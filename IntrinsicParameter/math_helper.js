/* math_helper.js
   Browser-only (single file). Provides:
   - Basic linear algebra helpers
   - SVD (Golub-Reinsch) (from your svd.js)
   - Jacobi EVD for symmetric matrices (from your jacobi.js)
   - DLT homography estimation (Hartley-normalized)
   - Zhang planar calibration (K from homographies) + extrinsics
   - Rodrigues / inverse Rodrigues
   - Simple Levenberg–Marquardt optimizer (numeric Jacobian)
   - calibratePlanarFromPoints(): end-to-end intrinsics + distortion + per-image extrinsics

   Usage:
     <script src="math_helper.js"></script>
     const result = MathHelper.calibratePlanarFromPoints(objPoints, images, { distModel: 'radial-tangential' });
*/

(function (global) {
  "use strict";

  /* ----------------------------- BASIC LINALG ----------------------------- */
  function zeros(r, c) {
    const A = new Array(r);
    for (let i = 0; i < r; i++) A[i] = new Array(c).fill(0);
    return A;
  }

  function cloneMat(A) {
    return A.map(row => row.slice());
  }

  function transpose(A) {
    const m = A.length, n = A[0].length;
    const T = zeros(n, m);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j];
    return T;
  }

  function matMul(A, B) {
    const m = A.length, n = B[0].length, k = B.length;
    const C = zeros(m, n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
        C[i][j] = s;
      }
    }
    return C;
  }

  function matVecMul(A, x) {
    const m = A.length, n = A[0].length;
    const y = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += A[i][j] * x[j];
      y[i] = s;
    }
    return y;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function norm2(a) {
    return Math.sqrt(dot(a, a));
  }

  function addVec(a, b) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }

  function subVec(a, b) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
  }

  function scaleVec(a, s) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
    return out;
  }

  function cross(a, b) {
    return [
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0],
    ];
  }

  function inv3x3(M) {
    const a = M[0][0], b = M[0][1], c = M[0][2];
    const d = M[1][0], e = M[1][1], f = M[1][2];
    const g = M[2][0], h = M[2][1], i = M[2][2];

    const A =  (e*i - f*h);
    const B = -(d*i - f*g);
    const C =  (d*h - e*g);
    const D = -(b*i - c*h);
    const E =  (a*i - c*g);
    const F = -(a*h - b*g);
    const G =  (b*f - c*e);
    const H = -(a*f - c*d);
    const I =  (a*e - b*d);

    const det = a*A + b*B + c*C;
    if (Math.abs(det) < 1e-12) throw new Error("inv3x3: singular matrix");

    const invDet = 1.0 / det;
    return [
      [A*invDet, D*invDet, G*invDet],
      [B*invDet, E*invDet, H*invDet],
      [C*invDet, F*invDet, I*invDet],
    ];
  }

  /* ------------------------------- SVD ------------------------------------ */
/** SVD procedure as explained in "Singular Value Decomposition and Least Squares Solutions. By G.H. Golub et al."
 *
 * This procedure computes the singular values and complete orthogonal decomposition of a real rectangular matrix A:
 *    A = U * diag(q) * V(t), U(t) * U = V(t) * V = I
 * where the arrays a, u, v, q represent A, U, V, q respectively. The actual parameters corresponding to a, u, v may
 * all be identical unless withu = withv = {true}. In this case, the actual parameters corresponding to u and v must
 * differ. m >= n is assumed (with m = a.length and n = a[0].length)
 *
 *  @param a {Array} Represents the matrix A to be decomposed
 *  @param [withu] {bool} {true} if U is desired {false} otherwise
 *  @param [withv] {bool} {true} if U is desired {false} otherwise
 *  @param [eps] {Number} A constant used in the test for convergence; should not be smaller than the machine precision
 *  @param [tol] {Number} A machine dependent constant which should be set equal to B/eps0 where B is the smallest
 *    positive number representable in the computer
 *
 *  @returns {Object} An object containing:
 *    q: A vector holding the singular values of A; they are non-negative but not necessarily ordered in
 *      decreasing sequence
 *    u: Represents the matrix U with orthonormalized columns (if withu is {true} otherwise u is used as
 *      a working storage)
 *    v: Represents the orthogonal matrix V (if withv is {true}, otherwise v is not used)
 *
 */
const SVD = (a, withu, withv, eps, tol) => {
  // Define default parameters
  withu = withu !== undefined ? withu : true
  withv = withv !== undefined ? withv : true
  eps = eps || Math.pow(2, -52)
  tol = 1e-64 / eps

  // throw error if a is not defined
  if (!a) {
    throw new TypeError('Matrix a is not defined')
  }

  // Householder's reduction to bidiagonal form

  const n = a[0].length
  const m = a.length

  if (m < n) {
    throw new TypeError('Invalid matrix: m < n')
  }

  let i, j, k, l, l1, c, f, g, h, s, x, y, z

  g = 0
  x = 0
  const e = []

  const u = []
  const v = []

  const mOrN = (withu === 'f') ? m : n

  // Initialize u
  for (i = 0; i < m; i++) {
    u[i] = new Array(mOrN).fill(0)
  }

  // Initialize v
  for (i = 0; i < n; i++) {
    v[i] = new Array(n).fill(0)
  }

  // Initialize q
  const q = new Array(n).fill(0)

  // Copy array a in u
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      u[i][j] = a[i][j]
    }
  }

  for (i = 0; i < n; i++) {
    e[i] = g
    s = 0
    l = i + 1
    for (j = i; j < m; j++) {
      s += Math.pow(u[j][i], 2)
    }
    if (s < tol) {
      g = 0
    } else {
      f = u[i][i]
      g = f < 0 ? Math.sqrt(s) : -Math.sqrt(s)
      h = f * g - s
      u[i][i] = f - g
      for (j = l; j < n; j++) {
        s = 0
        for (k = i; k < m; k++) {
          s += u[k][i] * u[k][j]
        }
        f = s / h
        for (k = i; k < m; k++) {
          u[k][j] = u[k][j] + f * u[k][i]
        }
      }
    }
    q[i] = g
    s = 0
    for (j = l; j < n; j++) {
      s += Math.pow(u[i][j], 2)
    }
    if (s < tol) {
      g = 0
    } else {
      f = u[i][i + 1]
      g = f < 0 ? Math.sqrt(s) : -Math.sqrt(s)
      h = f * g - s
      u[i][i + 1] = f - g
      for (j = l; j < n; j++) {
        e[j] = u[i][j] / h
      }
      for (j = l; j < m; j++) {
        s = 0
        for (k = l; k < n; k++) {
          s += u[j][k] * u[i][k]
        }
        for (k = l; k < n; k++) {
          u[j][k] = u[j][k] + s * e[k]
        }
      }
    }
    y = Math.abs(q[i]) + Math.abs(e[i])
    if (y > x) {
      x = y
    }
  }

  // Accumulation of right-hand transformations
  if (withv) {
    for (i = n - 1; i >= 0; i--) {
      if (g !== 0) {
        h = u[i][i + 1] * g
        for (j = l; j < n; j++) {
          v[j][i] = u[i][j] / h
        }
        for (j = l; j < n; j++) {
          s = 0
          for (k = l; k < n; k++) {
            s += u[i][k] * v[k][j]
          }
          for (k = l; k < n; k++) {
            v[k][j] = v[k][j] + s * v[k][i]
          }
        }
      }
      for (j = l; j < n; j++) {
        v[i][j] = 0
        v[j][i] = 0
      }
      v[i][i] = 1
      g = e[i]
      l = i
    }
  }

  // Accumulation of left-hand transformations
  if (withu) {
    if (withu === 'f') {
      for (i = n; i < m; i++) {
        for (j = n; j < m; j++) {
          u[i][j] = 0
        }
        u[i][i] = 1
      }
    }
    for (i = n - 1; i >= 0; i--) {
      l = i + 1
      g = q[i]
      for (j = l; j < mOrN; j++) {
        u[i][j] = 0
      }
      if (g !== 0) {
        h = u[i][i] * g
        for (j = l; j < mOrN; j++) {
          s = 0
          for (k = l; k < m; k++) {
            s += u[k][i] * u[k][j]
          }
          f = s / h
          for (k = i; k < m; k++) {
            u[k][j] = u[k][j] + f * u[k][i]
          }
        }
        for (j = i; j < m; j++) {
          u[j][i] = u[j][i] / g
        }
      } else {
        for (j = i; j < m; j++) {
          u[j][i] = 0
        }
      }
      u[i][i] = u[i][i] + 1
    }
  }

  // Diagonalization of the bidiagonal form
  eps = eps * x
  let testConvergence
  for (k = n - 1; k >= 0; k--) {
    for (let iteration = 0; iteration < 50; iteration++) {
      // test-f-splitting
      testConvergence = false
      for (l = k; l >= 0; l--) {
        if (Math.abs(e[l]) <= eps) {
          testConvergence = true
          break
        }
        if (Math.abs(q[l - 1]) <= eps) {
          break
        }
      }

      if (!testConvergence) { // cancellation of e[l] if l>0
        c = 0
        s = 1
        l1 = l - 1
        for (i = l; i < k + 1; i++) {
          f = s * e[i]
          e[i] = c * e[i]
          if (Math.abs(f) <= eps) {
            break // goto test-f-convergence
          }
          g = q[i]
          q[i] = Math.sqrt(f * f + g * g)
          h = q[i]
          c = g / h
          s = -f / h
          if (withu) {
            for (j = 0; j < m; j++) {
              y = u[j][l1]
              z = u[j][i]
              u[j][l1] = y * c + (z * s)
              u[j][i] = -y * s + (z * c)
            }
          }
        }
      }

      // test f convergence
      z = q[k]
      if (l === k) { // convergence
        if (z < 0) {
          // q[k] is made non-negative
          q[k] = -z
          if (withv) {
            for (j = 0; j < n; j++) {
              v[j][k] = -v[j][k]
            }
          }
        }
        break // break out of iteration loop and move on to next k value
      }

      // Shift from bottom 2x2 minor
      x = q[l]
      y = q[k - 1]
      g = e[k - 1]
      h = e[k]
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
      g = Math.sqrt(f * f + 1)
      f = ((x - z) * (x + z) + h * (y / (f < 0 ? (f - g) : (f + g)) - h)) / x

      // Next QR transformation
      c = 1
      s = 1
      for (i = l + 1; i < k + 1; i++) {
        g = e[i]
        y = q[i]
        h = s * g
        g = c * g
        z = Math.sqrt(f * f + h * h)
        e[i - 1] = z
        c = f / z
        s = h / z
        f = x * c + g * s
        g = -x * s + g * c
        h = y * s
        y = y * c
        if (withv) {
          for (j = 0; j < n; j++) {
            x = v[j][i - 1]
            z = v[j][i]
            v[j][i - 1] = x * c + z * s
            v[j][i] = -x * s + z * c
          }
        }
        z = Math.sqrt(f * f + h * h)
        q[i - 1] = z
        c = f / z
        s = h / z
        f = c * g + s * y
        x = -s * g + c * y
        if (withu) {
          for (j = 0; j < m; j++) {
            y = u[j][i - 1]
            z = u[j][i]
            u[j][i - 1] = y * c + z * s
            u[j][i] = -y * s + z * c
          }
        }
      }
      e[l] = 0
      e[k] = f
      q[k] = x
    }
  }

  // Number below eps should be zero
  for (i = 0; i < n; i++) {
    if (q[i] < eps) q[i] = 0
  }

  return { u, q, v }
}

  /* --------------------------- Jacobi EVD --------------------------------- */
// Rotation Matrix
var Rot = function(theta){
    var Mat = [[Math.cos(theta),Math.sin(theta)],[-Math.sin(theta),Math.cos(theta)]];
    return Mat
}
// Givens Matrix
var Rij = function(k,l,theta,N){
    var Mat = Array(N) 
    for (var i = 0; i<N;i++){
        Mat[i] = Array(N) 
    }
    // Identity Matrix
    for (var i = 0; i<N;i++){
        for (var j = 0; j<N;j++){
            Mat[i][j] = (i===j)*1.0;
        }
    }
    var Rotij = Rot(theta);

    // Put Rotation part in i, j
    Mat[k][k] = Rotij[0][0] // 11
    Mat[l][l] = Rotij[1][1] // 22
    Mat[k][l] = Rotij[0][1] // 12
    Mat[l][k] = Rotij[1][0] // 21
    return Mat
}

// get angle
var getTheta = function(aii,ajj,aij){
    var  th = 0.0 
    var denom = (ajj - aii);
    if (Math.abs(denom) <= 1E-12){
        th = Math.PI/4.0
    }
    else {
        th = 0.5 * Math.atan(2.0 * aij / (ajj - aii) ) 
    }
    return th 
}
// get max off-diagonal value from Upper Diagonal
var getAij = function(Mij){
    var N = Mij.length;
    var maxMij = 0.0 ;
    var maxIJ  = [0,1];
    for (var i = 0; i<N;i++){
        for (var j = i+1; j<N;j++){ 
            if (Math.abs(maxMij) <= Math.abs(Mij[i][j])){
                maxMij = Math.abs(Mij[i][j]);
                maxIJ  = [i,j];
            } 
        }
    }
    return [maxIJ,maxMij]
}
// Unitary Rotation UT x H x U
var unitary  = function(U,H){
    var N = U.length;
    // empty NxN matrix
    var Mat = Array(N) 
    for (var i = 0; i<N;i++){
        Mat[i] = Array(N) 
    }
    // compute element
    for (var i = 0; i<N;i++){
        for (var j = 0; j<N;j++){
            Mat[i][j] =  0 
            for (var k = 0; k<N;k++){
                for (var l = 0; l<N;l++){
                    Mat[i][j] = Mat[i][j] + U[k][i] * H[k][l] * U[l][j];
                }
            }
        }
    }
    return Mat;
}

// Matrix Multiplication
var AxB = function(A,B){
    var N = A.length;
    // empty NxN matrix
    var Mat = Array(N) 
    for (var i = 0; i<N;i++){
        Mat[i] = Array(N) 
    }
    for (var i = 0; i<N;i++){
        for (var j = 0; j<N;j++){
            Mat[i][j] =  0 
            for (var k = 0; k<N;k++){
                Mat[i][j] = Mat[i][j] + A[i][k] * B[k][j] ; 
            }
        }
    }
    return Mat;
}

var diag = function(Hij, convergence = 1E-7){
    var N = Hij.length; 
    var Ei = Array(N);
    var e0 =  Math.abs(convergence / N)
    // initial vector
    var Sij = Array(N);
    for (var i = 0; i<N;i++){
        Sij[i] = Array(N) 
    }
    // Sij is Identity Matrix
    for (var i = 0; i<N;i++){
        for (var j = 0; j<N;j++){
            Sij[i][j] = (i===j)*1.0;
        }
    }
    // initial error
    var Vab = getAij(Hij); 
    //  jacobi iterations
    while (Math.abs(Vab[1]) >= Math.abs(e0)){
        // block index to be rotated
        var i =  Vab[0][0];
        var j =  Vab[0][1];
        // get theta
        var psi = getTheta(Hij[i][i], Hij[j][j], Hij[i][j]); 
        // Givens matrix
        var Gij =  Rij(i,j,psi,N);
        // rotate Hamiltonian using Givens
        Hij = unitary(Gij,Hij); 
        // Update vectors
        Sij = AxB(Sij,Gij); 
        // update error 
        Vab = getAij(Hij); 
    }
    for (var i = 0; i<N;i++){
        Ei[i] = Hij[i][i]; 
    }
    return sorting(Ei , Sij) 
}


var sorting = function(E, S){
    var N = E.length ; 
    var Ef = Array(N);
    var Sf = Array(N);
    for (var k = 0; k<N;k++){
        Sf[k] = Array(N);
    }
    for (var i = 0; i<N;i++){
        var minID = 0;
        var minE  = E[0];
        for (var j = 0; j<E.length;j++){
            if (E[j] < minE){
                minID = j ; 
                minE  = E[minID];
            }
        }
        Ef[i] = E.splice(minID,1);
        for (var k = 0; k<N;k++){
            Sf[k][i]  = S[k][minID];
            S[k].splice(minID,1);
        }
    }
    return [Ef,Sf]
}

  function eigSymmetricJacobi(A, convergence = 1e-7) {
    const deep = JSON.parse(JSON.stringify(A));
    const out = diag(deep, convergence);
    const E = out[0].map(x => Array.isArray(x) ? x[0] : x);
    const U = out[1];
    return { E, U };
  }

  function nullspaceVector(A) {
    const { v: V } = SVD(A, true, true);
    return V.map(row => row[row.length - 1]);
  }

  /* ----------------------- Hartley Normalisierung (2D) --------------------- */
  function normalizePoints2D(pts) {
    const P = pts.map(p => Array.isArray(p) ? { x: p[0], y: p[1] } : p);
    const n = P.length;
    let mx = 0, my = 0;
    for (const p of P) { mx += p.x; my += p.y; }
    mx /= n; my /= n;

    let md = 0;
    for (const p of P) md += Math.sqrt((p.x - mx)**2 + (p.y - my)**2);
    md /= n;

    const s = Math.SQRT2 / (md || 1);

    const T = [
      [s, 0, -s * mx],
      [0, s, -s * my],
      [0, 0, 1],
    ];

    const npts = P.map(p => ({ x: s * (p.x - mx), y: s * (p.y - my) }));
    return { T, pts: npts };
  }

  /* ------------------------ DLT Homographie (planar) ----------------------- */
  function computeHomographyDLT(objPts, imgPts) {
    if (objPts.length !== imgPts.length) throw new Error("H DLT: length mismatch");
    if (objPts.length < 4) throw new Error("H DLT: need >= 4 points");

    const { T: To, pts: oN } = normalizePoints2D(objPts);
    const { T: Ti, pts: iN } = normalizePoints2D(imgPts);

    const A = [];
    for (let k = 0; k < oN.length; k++) {
      const X = oN[k].x, Y = oN[k].y;
      const u = iN[k].x, v = iN[k].y;
      A.push([-X, -Y, -1,  0,  0,  0,  u*X, u*Y, u]);
      A.push([ 0,  0,  0, -X, -Y, -1,  v*X, v*Y, v]);
    }

    const h = nullspaceVector(A);
    let Hn = [
      [h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], h[8]],
    ];

    const TiInv = inv3x3(Ti);
    Hn = matMul(matMul(TiInv, Hn), To);

    const s = Hn[2][2] || 1;
    return Hn.map(r => r.map(vv => vv / s));
  }

  /* ------------------------------- Zhang ---------------------------------- */
  function v_ij(H, i, j) {
    const hi0 = H[0][i], hi1 = H[1][i], hi2 = H[2][i];
    const hj0 = H[0][j], hj1 = H[1][j], hj2 = H[2][j];
    return [
      hi0*hj0,
      hi0*hj1 + hi1*hj0,
      hi1*hj1,
      hi2*hj0 + hi0*hj2,
      hi2*hj1 + hi1*hj2,
      hi2*hj2
    ];
  }

  function zhangComputeIntrinsicsFromHomographies(Hs) {
    if (!Hs || Hs.length < 2) throw new Error("Zhang: need at least 2 homographies (prefer 5+).");
    const V = [];
    for (const H of Hs) {
      const v12 = v_ij(H, 0, 1);
      const v11 = v_ij(H, 0, 0);
      const v22 = v_ij(H, 1, 1);
      V.push(v12);
      V.push(v11.map((x, idx) => x - v22[idx]));
    }
    const b = nullspaceVector(V); // [B11,B12,B22,B13,B23,B33] up to scale
    const B11 = b[0], B12 = b[1], B22 = b[2], B13 = b[3], B23 = b[4], B33 = b[5];

    const v0 = (B12*B13 - B11*B23) / (B11*B22 - B12*B12);
    const lambda = B33 - (B13*B13 + v0*(B12*B13 - B11*B23)) / B11;
    const alpha = Math.sqrt(lambda / B11);
    const beta = Math.sqrt(lambda * B11 / (B11*B22 - B12*B12));
    const gamma = -B12 * alpha * alpha * beta / lambda;
    const u0 = gamma * v0 / beta - (B13 * alpha * alpha) / lambda;

    const K = [
      [alpha, gamma, u0],
      [0,     beta,  v0],
      [0,     0,     1]
    ];
    return { K, b: [B11,B12,B22,B13,B23,B33] };
  }

  function orthonormalizeR(R) {
    const { u: U, v: V } = SVD(R, true, true);
    const Vt = transpose(V);
    let Rn = matMul(U, Vt);
    const det =
      Rn[0][0]*(Rn[1][1]*Rn[2][2]-Rn[1][2]*Rn[2][1]) -
      Rn[0][1]*(Rn[1][0]*Rn[2][2]-Rn[1][2]*Rn[2][0]) +
      Rn[0][2]*(Rn[1][0]*Rn[2][1]-Rn[1][1]*Rn[2][0]);
    if (det < 0) {
      for (let i = 0; i < 3; i++) U[i][2] *= -1;
      Rn = matMul(U, Vt);
    }
    return Rn;
  }

  function extrinsicsFromHomography(H, K) {
    const invK = inv3x3(K);
    const h1 = [H[0][0], H[1][0], H[2][0]];
    const h2 = [H[0][1], H[1][1], H[2][1]];
    const h3 = [H[0][2], H[1][2], H[2][2]];

    const invK_h1 = matVecMul(invK, h1);
    const invK_h2 = matVecMul(invK, h2);
    const invK_h3 = matVecMul(invK, h3);

    const lambda = 1.0 / norm2(invK_h1);
    const r1 = scaleVec(invK_h1, lambda);
    const r2 = scaleVec(invK_h2, lambda);
    const r3 = cross(r1, r2);
    const t  = scaleVec(invK_h3, lambda);

    let R = [
      [r1[0], r2[0], r3[0]],
      [r1[1], r2[1], r3[1]],
      [r1[2], r2[2], r3[2]],
    ];
    R = orthonormalizeR(R);
    return { R, t };
  }

  /* --------------------------- Rodrigues ---------------------------------- */
  function rodrigues(rvec) {
    const theta = norm2(rvec);
    if (theta < 1e-12) {
      return [
        [1,0,0],
        [0,1,0],
        [0,0,1],
      ];
    }
    const kx = rvec[0]/theta, ky = rvec[1]/theta, kz = rvec[2]/theta;
    const K = [
      [0, -kz, ky],
      [kz, 0, -kx],
      [-ky, kx, 0]
    ];
    const I = [
      [1,0,0],
      [0,1,0],
      [0,0,1],
    ];
    const K2 = matMul(K, K);
    const s = Math.sin(theta);
    const c = Math.cos(theta);
    const R = zeros(3,3);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      R[i][j] = I[i][j] + s*K[i][j] + (1-c)*K2[i][j];
    }
    return R;
  }

  function invRodrigues(R) {
    const tr = R[0][0] + R[1][1] + R[2][2];
    let cosTheta = (tr - 1) / 2;
    cosTheta = Math.min(1, Math.max(-1, cosTheta));
    const theta = Math.acos(cosTheta);
    if (theta < 1e-12) return [0,0,0];
    const denom = 2*Math.sin(theta);
    const rx = (R[2][1] - R[1][2]) / denom;
    const ry = (R[0][2] - R[2][0]) / denom;
    const rz = (R[1][0] - R[0][1]) / denom;
    return [rx*theta, ry*theta, rz*theta];
  }

  /* ------------------ Projection + distortion model ----------------------- */
  function projectPointPlanar(X, Y, rvec, tvec, K, dist) {
    const fx = K[0][0], skew = K[0][1], cx = K[0][2];
    const fy = K[1][1], cy = K[1][2];

    const k1 = dist[0] || 0, k2 = dist[1] || 0, k3 = dist[2] || 0;
    const p1 = dist[3] || 0, p2 = dist[4] || 0;

    const R = rodrigues(rvec);
    const Xc = R[0][0]*X + R[0][1]*Y + tvec[0];
    const Yc = R[1][0]*X + R[1][1]*Y + tvec[1];
    const Zc = R[2][0]*X + R[2][1]*Y + tvec[2];

    const x = Xc / Zc;
    const y = Yc / Zc;

    const r2 = x*x + y*y;
    const radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;

    let xd = x * radial;
    let yd = y * radial;

    if (p1 !== 0 || p2 !== 0) {
      xd += 2*p1*x*y + p2*(r2 + 2*x*x);
      yd += p1*(r2 + 2*y*y) + 2*p2*x*y;
    }

    const u = fx*xd + skew*yd + cx;
    const v = fy*yd + cy;
    return [u, v];
  }

  /* --------------------- Cholesky solve (SPD) ----------------------------- */
  function choleskyDecompose(A) {
    const n = A.length;
    const L = zeros(n,n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = A[i][j];
        for (let k = 0; k < j; k++) sum -= L[i][k]*L[j][k];
        if (i === j) {
          if (sum <= 0) return null;
          L[i][j] = Math.sqrt(sum);
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
    }
    return L;
  }

  function choleskySolve(L, b) {
    const n = L.length;
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let k = 0; k < i; k++) sum -= L[i][k]*y[k];
      y[i] = sum / L[i][i];
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i];
      for (let k = i + 1; k < n; k++) sum -= L[k][i]*x[k];
      x[i] = sum / L[i][i];
    }
    return x;
  }

  /* ---------------------- Levenberg–Marquardt ----------------------------- */
  function levenbergMarquardt(residualFunc, p0, opts = {}) {
    const maxIters = opts.maxIters ?? 30;
    const lambda0 = opts.lambda0 ?? 1e-3;
    const eps = opts.eps ?? 1e-6;
    const tol = opts.tol ?? 1e-6;
    const verbose = !!opts.verbose;

    let p = p0.slice();
    let lambda = lambda0;

    let r = residualFunc(p);
    let cost = dot(r, r);

    for (let iter = 0; iter < maxIters; iter++) {
      const m = r.length;
      const n = p.length;

      const J = zeros(m, n);
      for (let j = 0; j < n; j++) {
        const pj = p[j];
        const step = eps * (Math.abs(pj) + 1);
        p[j] = pj + step;
        const r1 = residualFunc(p);
        p[j] = pj - step;
        const r2 = residualFunc(p);
        p[j] = pj;

        const inv2s = 1.0 / (2*step);
        for (let i = 0; i < m; i++) {
          J[i][j] = (r1[i] - r2[i]) * inv2s;
        }
      }

      const JTJ = zeros(n, n);
      const JTr = new Array(n).fill(0);
      for (let i = 0; i < m; i++) {
        const ri = r[i];
        for (let j = 0; j < n; j++) {
          const Jij = J[i][j];
          JTr[j] += Jij * ri;
          for (let k = 0; k <= j; k++) {
            JTJ[j][k] += Jij * J[i][k];
          }
        }
      }
      for (let j = 0; j < n; j++) for (let k = 0; k < j; k++) JTJ[k][j] = JTJ[j][k];

      const A = cloneMat(JTJ);
      for (let j = 0; j < n; j++) A[j][j] += lambda * (JTJ[j][j] || 1);

      const L = choleskyDecompose(A);
      if (!L) { lambda *= 10; continue; }

      const dp = choleskySolve(L, scaleVec(JTr, -1));
      const pNew = addVec(p, dp);
      const rNew = residualFunc(pNew);
      const costNew = dot(rNew, rNew);

      if (costNew < cost) {
        p = pNew; r = rNew;
        const improvement = cost - costNew;
        cost = costNew;
        lambda = Math.max(lambda / 3, 1e-12);
        if (verbose) console.log("LM iter", iter, "cost", cost, "lambda", lambda);
        if (improvement / (cost + 1e-12) < tol) break;
      } else {
        lambda *= 5;
      }
    }
    return { p, cost };
  }

  /* --------------------- End-to-end planar calibration -------------------- */
  function calibratePlanarFromPoints(objPoints, images, options = {}) {
    // objPoints: array of {id,X,Y} (or {id,x,y}) in target plane units
    // images: array of {points:[{id,u,v}]} pixel coords, same ids
    const distModel = options.distModel ?? "radial-tangential";
    const optimizeSkew = options.optimizeSkew ?? false;

    const objMap = new Map();
    for (const p of objPoints) {
      const X = (p.X !== undefined) ? p.X : p.x;
      const Y = (p.Y !== undefined) ? p.Y : p.y;
      objMap.set(p.id, { X, Y });
    }

    const Hs = [];
    const perImg = [];
    for (const img of images) {
      const obj2D = [];
      const img2D = [];
      for (const q of img.points) {
        const o = objMap.get(q.id);
        if (!o) continue;
        obj2D.push({ x: o.X, y: o.Y });
        img2D.push({ x: q.u, y: q.v });
      }
      if (obj2D.length < 6) throw new Error("Need at least 6 matched points per image (better 10+).");
      const H = computeHomographyDLT(obj2D, img2D);
      Hs.push(H);
      perImg.push({ obj2D, img2D });
    }

    const { K: K0 } = zhangComputeIntrinsicsFromHomographies(Hs);

    const extr0 = Hs.map(H => extrinsicsFromHomography(H, K0));
    const rvecs0 = extr0.map(e => invRodrigues(e.R));
    const tvecs0 = extr0.map(e => e.t);

    const fx0 = K0[0][0], fy0 = K0[1][1], cx0 = K0[0][2], cy0 = K0[1][2], skew0 = K0[0][1];

    const p0 = [];
    p0.push(fx0, fy0, cx0, cy0);
    p0.push(optimizeSkew ? skew0 : 0);
    // distortion slots: k1,k2,k3,p1,p2 (p1/p2 only used if radial-tangential)
    p0.push(0,0,0,0,0);

    for (let i = 0; i < images.length; i++) {
      p0.push(rvecs0[i][0], rvecs0[i][1], rvecs0[i][2]);
      p0.push(tvecs0[i][0], tvecs0[i][1], tvecs0[i][2]);
    }

    function unpack(pp) {
      let idx = 0;
      const fx = pp[idx++], fy = pp[idx++], cx = pp[idx++], cy = pp[idx++];
      const skew = pp[idx++];
      const k1 = pp[idx++], k2 = pp[idx++], k3 = pp[idx++], p1 = pp[idx++], p2 = pp[idx++];

      const K = [
        [fx, optimizeSkew ? skew : 0, cx],
        [0,  fy,               cy],
        [0,  0,                1]
      ];
      const dist = (distModel === "radial-tangential") ? [k1,k2,k3,p1,p2] : [k1,k2,k3,0,0];

      const rvecs = [];
      const tvecs = [];
      for (let i = 0; i < images.length; i++) {
        rvecs.push([pp[idx++], pp[idx++], pp[idx++]]);
        tvecs.push([pp[idx++], pp[idx++], pp[idx++]]);
      }
      return { K, dist, rvecs, tvecs };
    }

    function residual(pp) {
      const { K, dist, rvecs, tvecs } = unpack(pp);
      const res = [];
      for (let i = 0; i < images.length; i++) {
        const o2D = perImg[i].obj2D;
        const im2D = perImg[i].img2D;
        for (let k = 0; k < o2D.length; k++) {
          const X = o2D[k].x, Y = o2D[k].y;
          const [up, vp] = projectPointPlanar(X, Y, rvecs[i], tvecs[i], K, dist);
          res.push(up - im2D[k].x);
          res.push(vp - im2D[k].y);
        }
      }
      return res;
    }

    const lm = levenbergMarquardt(residual, p0, {
      maxIters: options.maxIters ?? 25,
      lambda0: options.lambda0 ?? 1e-3,
      eps: options.jacEps ?? 1e-6,
      tol: options.tol ?? 1e-6,
      verbose: !!options.verbose
    });

    const { K, dist, rvecs, tvecs } = unpack(lm.p);
    const rFinal = residual(lm.p);
    const rms = Math.sqrt(dot(rFinal, rFinal) / (rFinal.length || 1));

    return { K, dist, rvecs, tvecs, rms, homographies: Hs };
  }

  /* ----------------------------- Public API ------------------------------- */
  global.MathHelper = {
    // linalg
    zeros, cloneMat, transpose, matMul, matVecMul, dot, norm2, addVec, subVec, scaleVec, cross, inv3x3,
    // svd/evd
    SVD, nullspaceVector, eigSymmetricJacobi,
    // homography
    normalizePoints2D, computeHomographyDLT,
    // zhang
    zhangComputeIntrinsicsFromHomographies, extrinsicsFromHomography,
    // rotations
    rodrigues, invRodrigues,
    // projection
    projectPointPlanar,
    // optimizer
    levenbergMarquardt,
    // full calibration
    calibratePlanarFromPoints
  };

})(window);
