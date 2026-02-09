/**
 * Real-world Target Generator / Input Calculation
 *
 * Dieses Modul erzeugt ein "asymmetrisches"/versetztes Punkt-Grid (planar):
 *   x = (2*i + (j mod 2)) * pitchX
 *   y = j * pitchY
 *
 * WICHTIG: In diesem asymmetrischen Grid ist der horizontale Nachbar-Abstand
 * (same row, next point) = 2 * pitchX.
 *
 * Um Druck-/Skalierungsfehler zu umgehen, kann der User statt pitchMm direkt messen:
 *  1) horizontaler Nachbar-Abstand (same row, next point)  -> pitchX = val / 2
 *  2) vertikaler Nachbar-Abstand   (next row, same column) -> pitchY = val
 *
 * Die API bleibt rückwärtskompatibel: Wenn nur pitchMm angegeben ist, wird pitchX=pitchY=pitchMm verwendet.
 */

/**
 * Ableitung von pitchX/pitchY aus unterschiedlichen Eingaben.
 *
 * @param {object} params
 * @param {number=} params.pitchMm - legacy: Pitch in mm (wird als pitchX/pitchY verwendet)
 * @param {number=} params.measuredHorizontalNeighborMm - gemessener Abstand zum nächsten Punkt rechts (same row, next point)
 * @param {number=} params.measuredVerticalNeighborMm - gemessener Abstand zum nächsten Punkt darunter (next row, same column)
 * @param {number=} params.pitchX - optional direkt gesetzt (überschreibt Berechnung)
 * @param {number=} params.pitchY - optional direkt gesetzt (überschreibt Berechnung)
 * @returns {{ pitchX: number, pitchY: number }}
 */
export function derivePitchXY({
  pitchMm,
  measuredHorizontalNeighborMm,
  measuredVerticalNeighborMm,
  pitchX,
  pitchY
} = {}) {
  const base = (typeof pitchMm === "number" && isFinite(pitchMm) && pitchMm > 0) ? pitchMm : undefined;

  // pitchX: bevorzugt direkt gesetzt, sonst aus horizontaler Messung, sonst legacy pitchMm
  let px = (typeof pitchX === "number" && isFinite(pitchX) && pitchX > 0) ? pitchX : undefined;
  if (px === undefined) {
    if (typeof measuredHorizontalNeighborMm === "number" && isFinite(measuredHorizontalNeighborMm) && measuredHorizontalNeighborMm > 0) {
      px = measuredHorizontalNeighborMm / 2;
    } else if (base !== undefined) {
      px = base;
    }
  }

  // pitchY: bevorzugt direkt gesetzt, sonst aus vertikaler Messung, sonst legacy pitchMm
  let py = (typeof pitchY === "number" && isFinite(pitchY) && pitchY > 0) ? pitchY : undefined;
  if (py === undefined) {
    if (typeof measuredVerticalNeighborMm === "number" && isFinite(measuredVerticalNeighborMm) && measuredVerticalNeighborMm > 0) {
      py = measuredVerticalNeighborMm;
    } else if (base !== undefined) {
      py = base;
    }
  }

  if (px === undefined || py === undefined) {
    throw new Error(
      "derivePitchXY: Bitte entweder pitchMm setzen oder measuredHorizontalNeighborMm+measuredVerticalNeighborMm (oder pitchX/pitchY)."
    );
  }

  return { pitchX: px, pitchY: py };
}

/**
 * Erzeugt ein asymmetrisches/versetztes Grid im "grid space" (noch nicht aufs Papier gelegt).
 *
 * @param {object} params
 * @param {number} params.cols
 * @param {number} params.rows
 * @param {number=} params.pitchMm - legacy: Pitch in mm (pitchX=pitchY)
 * @param {number=} params.measuredHorizontalNeighborMm - gemessener horizontaler Nachbar-Abstand (same row, next point)
 * @param {number=} params.measuredVerticalNeighborMm - gemessener vertikaler Nachbar-Abstand (next row, same column)
 * @param {number=} params.pitchX - optional direkt gesetzt
 * @param {number=} params.pitchY - optional direkt gesetzt
 * @returns {Array<{index:number,i:number,j:number,x:number,y:number}>}
 */
export function generateAsymGridPoints({
  cols,
  rows,
  pitchMm,
  measuredHorizontalNeighborMm,
  measuredVerticalNeighborMm,
  pitchX,
  pitchY
}) {
  const { pitchX: px, pitchY: py } = derivePitchXY({
    pitchMm,
    measuredHorizontalNeighborMm,
    measuredVerticalNeighborMm,
    pitchX,
    pitchY
  });

  const pts = [];
  let idx = 0;

  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      pts.push({
        index: idx++,
        i,
        j,
        x: (2 * i + (j & 1)) * px,
        y: j * py
      });
    }
  }
  return pts;
}

/**
 * Legt Punkte aufs Blatt und zentriert sie in den "usable bounds".
 * Berücksichtigt Margin und optional einen Kreisradius (damit die Kreise nicht in den Rand ragen).
 *
 * @param {Array<{x:number,y:number}>} pts Grid-Punkte in mm
 * @returns {{ placedPoints: Array, offsetX: number, offsetY: number, fits: boolean }}
 *          placedPoints: gleiche Punkte + pageX/pageY
 */
export function placePointsOnPaper({
  pts,
  paperWmm,
  paperHmm,
  marginMm = 0,
  circleDiameterMm = 0
}) {
  if (!pts || pts.length === 0) {
    return { placedPoints: [], offsetX: 0, offsetY: 0, fits: true };
  }

  const r = circleDiameterMm > 0 ? circleDiameterMm / 2 : 0;

  // Bounding box der Punkte (nur Zentren)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }

  // "Grid size" als Bounding-Box inkl. Kreisradius
  const gridW = (maxX - minX) + 2 * r;
  const gridH = (maxY - minY) + 2 * r;

  // Nutzbarer Bereich auf dem Blatt
  const usableW = paperWmm - 2 * marginMm;
  const usableH = paperHmm - 2 * marginMm;

  const fits = gridW <= usableW && gridH <= usableH;

  // Zentrier-Offset im nutzbaren Bereich, plus Margin
  // Zusätzlich: -minX/-minY, damit wir bei 0 starten, und +r, damit Kreis sauber berücksichtigt ist.
  const offsetX = marginMm + (usableW - gridW) / 2 + r - minX;
  const offsetY = marginMm + (usableH - gridH) / 2 + r - minY;

  const placedPoints = pts.map(p => ({
    ...p,
    pageX: p.x + offsetX,
    pageY: p.y + offsetY
  }));

  return { placedPoints, offsetX, offsetY, fits };
}

/**
 * Convenience: direkt Grid generieren + auf Papier platzieren
 *
 * @param {object} params
 * @param {number} params.cols
 * @param {number} params.rows
 * @param {number=} params.pitchMm - legacy (pitchX=pitchY)
 * @param {number=} params.measuredHorizontalNeighborMm - gemessener horizontaler Nachbar-Abstand (same row, next point)
 * @param {number=} params.measuredVerticalNeighborMm - gemessener vertikaler Nachbar-Abstand (next row, same column)
 * @param {number=} params.pitchX - optional direkt gesetzt
 * @param {number=} params.pitchY - optional direkt gesetzt
 * @param {number} params.paperWmm
 * @param {number} params.paperHmm
 * @param {number=} params.marginMm
 * @param {number=} params.circleDiameterMm
 */
export function generateAndPlace({
  cols,
  rows,
  pitchMm,
  measuredHorizontalNeighborMm,
  measuredVerticalNeighborMm,
  pitchX,
  pitchY,
  paperWmm,
  paperHmm,
  marginMm = 0,
  circleDiameterMm = 0
}) {
  const gridPts = generateAsymGridPoints({
    cols,
    rows,
    pitchMm,
    measuredHorizontalNeighborMm,
    measuredVerticalNeighborMm,
    pitchX,
    pitchY
  });

  return placePointsOnPaper({
    pts: gridPts,
    paperWmm,
    paperHmm,
    marginMm,
    circleDiameterMm
  });
}
