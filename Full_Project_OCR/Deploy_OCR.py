import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from spellchecker import SpellChecker
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import digit_version

### Text Detections
TEXT_DET_CUSTOM = {
    "config": "C:\\Users\\bobse\\Downloads\\Dibimbing-AI & ML Bootcamp\\Day 25 - Optical Character Recognition (OCR)\\All-about-OCR\\Full_Project_OCR\\dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py"
}

### Text Recognition
TEXT_RECOG_CUSTOM = {
    "config": "C:\\Users\\bobse\\Downloads\\Dibimbing-AI & ML Bootcamp\\Day 25 - Optical Character Recognition (OCR)\\All-about-OCR\\Full_Project_OCR\\svtr-base_20e_st_mj.py"
}

IMAGE_TEST = "C:\\Users\\bobse\\Downloads\\Dibimbing-AI & ML Bootcamp\\Day 25 - Optical Character Recognition (OCR)\\All-about-OCR\\Full_Project_OCR\\Training\\4.jpg"
DET_MODEL = TEXT_DET_CUSTOM

img = mmcv.imread(IMAGE_TEST, channel_order="rgb")
inferencer = TextDetInferencer(model=DET_MODEL["config"], weights=DET_MODEL["weights"])
det_result = inferencer(img)
polys_raw = det_result["predictions"][0]["polygons"]

polys = []
for poly in polys_raw:
  c = np.array(poly).astype(int).reshape((-1, 2))
  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  polys.append(box)

def organize_points(rect):
  """
  Sort 4 vertices polygon into the same order
  [top-left, top-right, bottom-right, bottom-left]
  """
  # sort points based on the x coordinate
  # so the order now [left-1, left-2, right-1, right-2]
  # we don't know which one is top or bottom yet
  points = sorted(list(rect), key=lambda x: x[0])
  # for 2 left points
  # the point with smaller y become top-left
  if points[1][1] > points[0][1]:
      index_1 = 0
      index_4 = 1
  else:
      index_1 = 1
      index_4 = 0
  # for 2 right points
  # the point with smaller y become top-right
  if points[3][1] > points[2][1]:
      index_2 = 2
      index_3 = 3
  else:
      index_2 = 3
      index_3 = 2

  return np.array([
      points[index_1], points[index_2], points[index_3], points[index_4]
  ])

def extract_text_image(rect, img):
  """
  Given 4 vertices polygon, crop the image using that
  polygon, and transform the crop so that it is flat.
  """
  # get the top-left, top-right, bottom-right, and bottom-left points.
  rect = organize_points(rect)
  tl, tr, br, bl = rect

  # determine the flattened image width,
  # which is equal to the largest bottom/top side of the polygon
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  # determine the flattened image height,
  # which is equal to the largest left/right side of the polygon
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  # create the 4 vertices polygon after transformation
  dst = np.array(
    [
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1],
    ],
    dtype=np.float32,
  )
  # do the transformation
  M = cv2.getPerspectiveTransform(rect, dst)
  return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

REC_MODEL = TEXT_RECOG_CUSTOM
inferencer = TextRecInferencer(
    model="/content/gdrive/MyDrive/TextRecognition_svtr_training/TextRecognition_svtr_training/svtr-base_20e_st_mj.py",
    weights="/content/gdrive/MyDrive/TextRecognition_svtr_training/TextRecognition_svtr_training/epoch_20.pth")

text_images = []
texts = []
for poly in polys:
  poly_arr = np.array(poly).reshape((-1, 2)).astype(np.float32)
  txt_img = extract_text_image(poly_arr, img)
  rec_result = inferencer(txt_img)
  text_images.append(txt_img)
  texts.append(rec_result["predictions"][0]["text"])

texts = [value.replace('<UKN>', '') if isinstance(value, str) and value != '<UKN>' else '' for value in texts]
texts = list(filter(None, texts))

def spell_check_result(ocr_result):
    # Create a SpellChecker instance
    spell = SpellChecker()

    # Join the OCR result list into a space-separated string
    text = ' '.join(ocr_result)

    # Tokenize the text into words
    words = text.split()

    # Perform spell checking on each word, excluding None values
    corrected_words = [spell.correction(word) for word in words if word and spell.correction(word)]

    # Join the corrected words into a space-separated string
    corrected_text = ' '.join(corrected_words)

    return corrected_text

spell_check_result(texts)


