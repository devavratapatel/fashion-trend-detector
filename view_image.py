import cv2
import pandas as pd
import os

df = pd.read_csv("labeled.csv")

for i in range(10):
    img_path = df.loc[i, "image"]
    print("img_path =", img_path)
    print("type(img_path) =", type(img_path))

    if pd.isna(img_path):
        print("❌ Image path is empty (NaN).")
    elif not os.path.exists(img_path):
        print(f"❌ File does not exist: {img_path}")
    else:
        image = cv2.imread(img_path)
        if image is not None:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"❌ Failed to read image: {img_path}")