let width = 0;
let height = 0;
// whether streaming video from the camera.
// Import keras model
let streaming = false;
let video = document.getElementById("video");
let stream = null;
let cap = null;
let src = null;
let dst = null;
let output = null;
let corners = null;
let processedDst = null;
let croppedDst = null;
let puzzleNotFound = 0;
let puzzleSolved = false;
function startCamera() {
  if (streaming) return;
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then(function (s) {
      stream = s;
      video.srcObject = s;
      video.play();
    })
    .catch(function (err) {
      console.log("An error occured! " + err);
    });

  video.addEventListener(
    "canplay",
    function (ev) {
      if (!streaming) {
        height = video.videoHeight;
        width = video.videoWidth;
        video.setAttribute("width", width);
        video.setAttribute("height", height);
        streaming = true;
        cap = new cv.VideoCapture(video);
      }
      startVideoProcessing();
    },
    false
  );
}
async function tensorFlow() {
  const model = await tf.loadLayersModel("resources/model.json");
}
function startVideoProcessing() {
  tensorFlow();
  if (!streaming) {
    console.warn("Please startup your webcam");
    return;
  }
  stopVideoProcessing();
  src = new cv.Mat(height, width, cv.CV_8UC4);
  dst = new cv.Mat(height, width, cv.CV_8UC4);
  processedDst = new cv.Mat(height, width, cv.CV_8UC4);
  requestAnimationFrame(processVideo);
}
function processVideo() {
  cap.read(src);
  let result = src;
  let processed = processImage(src);
  let corners = findCorners(processed);
  if (corners != null) {
    puzzleNotFound = 0;
    if (!puzzleSolved) {
      let cropped = cropPuzzle(processed, corners);
      board = extractBoard(cropped);
      if (board != null) {
        //Solve puzzle
        puzzleSolved = true;
      }
    }
  } else {
    puzzleNotFound += 1;
    if (puzzleNotFound > 10) {
      puzzleSolved = false;
    }
  }
  if (puzzleSolved) {
    result = overlayPuzzle(src, board, dst.rows, corners);
  }
  cv.imshow("canvasOutput", result);
  requestAnimationFrame(processVideo);
}
function stopVideoProcessing() {
  if (src != null && !src.isDeleted()) src.delete();
  if (dst != null && !dst.isDeleted()) dstC1.delete();
}
function opencvIsReady() {
  console.log("OpenCV.js is ready");
  info.innerHTML = "OpenCV.js is ready";
  startCamera();
}

function processImage(src) {
  // Grayscale
  cv.cvtColor(src, processedDst, cv.COLOR_RGBA2GRAY);

  // Guassian filter
  let ksize = new cv.Size(5, 5);
  cv.GaussianBlur(processedDst, processedDst, ksize, 0, 0, cv.BORDER_DEFAULT);
  ksize.delete;

  // Threshold
  cv.adaptiveThreshold(
    processedDst,
    processedDst,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    7,
    2
  );
  return processedDst;
}

function findCorners(src) {
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(
    src,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );

  // Find index of largest contour by area
  if (contours.size() > 0) {
    let maxIndex = 0;
    let area = 0;
    let tempArea;
    for (let i = 0; i < contours.size(); i++) {
      tempArea = cv.contourArea(contours.get(i), false);
      if (tempArea > area) {
        maxIndex = i;
        area = tempArea;
      }
    }
    area.delete;
    tempArea.delete;

    // Check contour is roughly square
    let perimeter = cv.arcLength(contours.get(maxIndex), true);
    let approx = new cv.Mat();
    cv.approxPolyDP(contours.get(maxIndex), approx, 0.02 * perimeter, true);
    perimeter.delete;

    // If approx has 4 corners, assume puzzle found for now
    if (approx.rows == 4) {
      approx.delete();
      // Find corners of largest contour representing the sudoku grid
      //Top left should have the smallest (x + y) value
      // Top right has largest (x - y) value
      // Bottom right corner of puzzle will have largest (x + y) value
      // Bottom left has smallest (x - y) value
      // return np.array([top_left, top_right, bottom_right, bottom_left]), processed_image

      // Array for holding 4 corner points of the puzzle
      let corners = [9999, 9999, 0, 0, 0, 0, 9999, 9999];

      for (let i = 0; i < contours.get(maxIndex).data32S.length; i += 2) {
        // Top left
        if (
          contours.get(maxIndex).data32S[i] +
            contours.get(maxIndex).data32S[i + 1] <
          corners[0] + corners[1]
        ) {
          corners[0] = contours.get(maxIndex).data32S[i];
          corners[1] = contours.get(maxIndex).data32S[i + 1];
        }
        // Top right
        if (
          contours.get(maxIndex).data32S[i] -
            contours.get(maxIndex).data32S[i + 1] >
          corners[2] - corners[3]
        ) {
          corners[2] = contours.get(maxIndex).data32S[i];
          corners[3] = contours.get(maxIndex).data32S[i + 1];
        }
        // Bottom right
        if (
          contours.get(maxIndex).data32S[i] +
            contours.get(maxIndex).data32S[i + 1] >
          corners[4] + corners[5]
        ) {
          corners[4] = contours.get(maxIndex).data32S[i];
          corners[5] = contours.get(maxIndex).data32S[i + 1];
        }
        // Bottom left
        if (
          contours.get(maxIndex).data32S[i] -
            contours.get(maxIndex).data32S[i + 1] <
          corners[6] - corners[7]
        ) {
          corners[6] = contours.get(maxIndex).data32S[i];
          corners[7] = contours.get(maxIndex).data32S[i + 1];
        }
      }
      contours.delete();
      hierarchy.delete();
      return corners;
    }
    approx.delete();
  }
  contours.delete();
  hierarchy.delete();
  return null;
}

function cropPuzzle(src, corners) {
  // Set side to length from top left to top right corner, use for warped image
  let a = corners[0] - corners[2];
  let b = corners[1] - corners[3];
  let side = Math.sqrt(Math.pow(a, 2) + Math.pow(b, 2));

  croppedDst = new cv.Mat(side, side, cv.CV_8UC1);

  let dsize = new cv.Size(side, side);
  let pt1 = cv.matFromArray(4, 1, cv.CV_32FC2, corners);
  let pt2 = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0,
    0,
    side,
    0,
    side,
    side,
    0,
    side,
  ]);
  let M = cv.getPerspectiveTransform(pt1, pt2);
  cv.warpPerspective(
    src,
    croppedDst,
    M,
    dsize,
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar()
  );
  dsize.delete;
  M.delete();
  pt1.delete();
  pt2.delete();
  return croppedDst;
}

// Sorting functions
function swap(items, leftIndex, rightIndex) {
  var temp = items[leftIndex];
  items[leftIndex] = items[rightIndex];
  items[rightIndex] = temp;
}
function partition(items, left, right) {
  var pivot = cv.boundingRect(items[Math.floor((right + left) / 2)]).x; //middle element
  var i = left; //left pointer
  var j = right; //right pointer
  while (i <= j) {
    while (cv.boundingRect(items[i]).x < pivot) {
      i++;
    }
    while (cv.boundingRect(items[j]).x > pivot) {
      j--;
    }

    if (i <= j) {
      swap(items, i, j); //sawpping two elements
      i++;
      j--;
    }
  }
  return i;
}

function quickSort(items, left, right) {
  var index;
  if (items.length > 1) {
    index = partition(items, left, right); //index returned from partition
    if (left < index - 1) {
      //more elements on the left side of the pivot
      quickSort(items, left, index - 1);
    }
    if (index < right) {
      //more elements on the right side of the pivot
      quickSort(items, index, right);
    }
  }
  return items;
}

function extractBoard(src) {
  let extractDst = new cv.Mat(src.rows, src.cols, cv.CV_8UC4);
  // Init board
  let board = [
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ],
  ];

  // Extract cells from board
  // Calculate estimate for what each cell area should be
  let cellAreaEst = Math.pow(src.rows / 9, 2);
  let limit = cellAreaEst / 5;
  // Loop through different kernel sizes to close lines
  for (let i = 5; i < 12; i += 2) {
    // Close horizontal and vertical lines
    let kernel = new cv.Mat();
    ksize = new cv.Size(i, i);
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize);
    cv.morphologyEx(src, extractDst, cv.MORPH_CLOSE, kernel);
    kernel.delete();
    ksize.delete;

    // Invert image so its white on black
    cv.bitwise_not(extractDst, extractDst);

    // Find outline contours
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(
      extractDst,
      contours,
      hierarchy,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE
    );
    var contoursInRange = new cv.MatVector();
    let tempArea;
    for (let i = 0; i < contours.size(); i++) {
      tempArea = cv.contourArea(contours.get(i), false);
      if (tempArea > cellAreaEst - limit && tempArea < cellAreaEst + limit) {
        contoursInRange.push_back(contours.get(i));
      }
    }
    contours.delete();
    hierarchy.delete();

    if (contoursInRange.size() == 81) {
      break;
    } else {
      if (i == 11) {
        contoursInRange.delete();
        return null;
      } else {
        continue;
      }
    }
  }
  // Sort contours into top to bottom
  let contoursSortedVertical = new cv.MatVector();
  for (let i = contoursInRange.size() - 1; i > -1; i--) {
    contoursSortedVertical.push_back(contoursInRange.get(i));
  }
  contoursInRange.delete();

  // Sort contours into left to right
  let contoursSortedHorizontal = new cv.MatVector();
  let temp;
  for (let i = 0; i < 9; i++) {
    temp = [];
    for (let j = 0; j < 9; j++) {
      temp.push(contoursSortedVertical.get(i * 9 + j));
    }
    // Sort
    temp = quickSort(temp, 0, temp.length - 1);
    for (let j = 0; j < 9; j++) {
      contoursSortedHorizontal.push_back(temp[j]);
    }
  }
  contoursSortedVertical.delete();

  // Fill board
  for (let j = 0; j < 9; j++) {
    for (let i = 0; i < 9; i++) {
      let rect = cv.boundingRect(contoursSortedHorizontal.get(j * 9 + i));
      cell = src.roi(rect);
      cell = identifyCell(cell, cv);
      if (cell != null) {
        board[j][i][1] = 1;
        cell.delete();
      }
    }
  }
  contoursSortedHorizontal.delete();
  extractDst.delete();
  return board;
}

function identifyCell(cell) {
  let w = cell.cols;
  let h = cell.rows;
  let x = Math.floor(w * 0.1);
  let y = Math.floor(h * 0.1);
  //remove outer 10% of cell
  let rect = new cv.Rect(x, y, w - x, h - y);
  cell = cell.roi(rect);
  // Find contours
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(
    cell,
    contours,
    hierarchy,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE
  );
  if (contours.size() == 0) {
    return null;
  }
  // Find index of largest contour
  let maxIndex = 0;
  let area = 0;
  let tempArea;
  for (let i = 0; i < contours.size(); i++) {
    tempArea = cv.contourArea(contours.get(i), false);
    if (tempArea > area) {
      maxIndex = i;
      area = tempArea;
    }
  }
  area.delete;
  tempArea.delete;

  // Create mask
  let mask = cv.Mat.zeros(cell.rows, cell.cols, cv.CV_8U);
  // Draw largest contour on mask
  cv.drawContours(mask, contours, maxIndex, new cv.Scalar(255), -1);

  if (cv.countNonZero(mask) / (cell.rows * cell.cols) < 0.05) {
    return null;
  }

  // Check if contour is too close to the edge, indicating noise
  rect = cv.boundingRect(contours.get(maxIndex));
  let limitX = cell.cols * 0.1;
  let limitY = cell.rows * 0.1;
  if (
    rect.x < limitX ||
    rect.y < limitY ||
    rect.x + rect.w > cell.cols - limitX ||
    rect.y + rect.h > cell.rows - limitY
  ) {
    return null;
  }
  // Apply mask to initial cell
  cv.bitwise_and(cell, mask, cell);
  mask.delete();

  // Final crop to make square
  w = cell.cols;
  h = cell.rows;
  if (w > h) {
    rect = new cv.Rect(
      Math.floor((w - h) / 2),
      0,
      Math.floor((w - h) / 2) + h,
      h
    );
    cell = cell.roi(rect);
  } else {
    rect = new cv.Rect(
      0,
      Math.floor((h - w) / 2),
      w,
      Math.floor((h - w) / 2) + w
    );
    cell = cell.roi(rect);
  }

  return cell;
}

function overlayPuzzle(src, board, gridSize, corners) {
  let overlay = new cv.Mat(
    gridSize,
    gridSize,
    cv.CV_8UC4,
    new cv.Scalar(0, 0, 0)
  );
  try {
    // Draw gridlines
    cellSize = Math.floor(gridSize / 9);
    for (let i = 0; i < 10; i++) {
      cv.line(
        overlay,
        new cv.Point(0, i * cellSize),
        new cv.Point(gridSize, i * cellSize),
        [0, 255, 0, 255],
        2
      );
      cv.line(
        overlay,
        new cv.Point(i * cellSize, 0),
        new cv.Point(i * cellSize, gridSize),
        [0, 255, 0, 255],
        2
      );
    }
    // Draw digits onto overlay
    let font = cv.FONT_HERSHEY_SIMPLEX;
    let scale = cellSize / 50;
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        if (board[j][i][1] == 0) {
          let text = board[j][i][0].toString();
          cv.putText(
            overlay,
            text,
            new cv.Point(i * cellSize, j * cellSize + cellSize),
            font,
            scale,
            [0, 255, 0, 255],
            2
          );
        }
      }
    }
    // Warp overlay
    let dsize = new cv.Size(src.cols, src.rows);
    let pt1 = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0,
      0,
      gridSize,
      0,
      gridSize,
      gridSize,
      0,
      gridSize,
    ]);
    let pt2 = cv.matFromArray(4, 1, cv.CV_32FC2, corners);
    let M = cv.getPerspectiveTransform(pt1, pt2);
    cv.warpPerspective(
      overlay,
      overlay,
      M,
      dsize,
      cv.INTER_LINEAR,
      cv.BORDER_CONSTANT,
      new cv.Scalar()
    );
    // Add overlay to source
    let addedImage = new cv.Mat(src.rows, src.cols, cv.CV_8UC4);
    cv.addWeighted(src, 1, overlay, 1, 0, addedImage);
    overlay.delete();
    return addedImage;
  } catch (err) {
    overlay.delete();
    return src;
  }
}
