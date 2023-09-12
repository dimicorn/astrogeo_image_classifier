from image import Image


def main():
    im = Image()
    obj = im.get_objects()[0]
    im.draw_uv(obj)
    im.draw_map(obj)


if __name__ == "__main__":
    main()
