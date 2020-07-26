from PIL import Image

IMAGE_NAME = "prod_support_logo"
IMAGE_TYPE = "png"
ORIGINAL_IMAGE_PATH = f"converted/{IMAGE_NAME}.{IMAGE_TYPE}"
ICON_PATH = f"converted/icons/{IMAGE_NAME}.ico"

img = Image.open(ORIGINAL_IMAGE_PATH)
img.save(ICON_PATH,format = 'ICO', sizes=[(32,32)])
