let src = new cv.Mat(height, width, cv.CV_8UC4);
let dst = new cv.Mat(height, width, cv.CV_8UC1);
let cap = new cv.VideoCapture(videoSource);
const FPS = 30;
function processVideo() {
  let begin = Date.now();
  cap.read(src);
  cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
  cv.imshow("canvasOutput", dst);
  // schedule next one.
  let delay = 1000 / FPS - (Date.now() - begin);
  setTimeout(processVideo, delay);
}
// schedule first one.
setTimeout(processVideo, 0);
imgElement.onload = function () {
  let mat = cv.imread(imgElement);
  cv.imshow("canvasOutput", mat);
  mat.delete();
};
function onOpenCvReady() {
  document.getElementById("status").innerHTML = "OpenCV.js is ready.";
}
