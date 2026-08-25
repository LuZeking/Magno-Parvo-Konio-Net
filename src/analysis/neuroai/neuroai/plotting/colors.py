"""
Define color scheme.
"""


def rgb(r, g, b, divide_by=255.0):
    """Convenience function: return colour in [0, 1]."""
    return (r/divide_by, g/divide_by, b/divide_by)


# primary colors
red = rgb(165, 30, 55)
gold = rgb(180, 160, 105)
metallic = rgb(50, 65, 75)

# secondary colors
blue1 = rgb(65, 90, 140)
blue2 = rgb(0, 105, 170)
blue3 = rgb(80, 170, 200)

green1 = rgb(50, 110, 30)
green2 = rgb(125, 165, 75)
green3 = rgb(130, 185, 160)

grey1 = rgb(180, 160, 150)
purple1 = rgb(175, 110, 150)
red1 = rgb(200, 80, 60)

brown1 = rgb(145, 105, 70)
orange1 = rgb(210, 150, 0)
orange2 = rgb(215, 180, 105)

# other colors
black = rgb(0, 0, 0)

# grayscale
white = rgb(255, 255, 255)  # 100% gray, white
grey1 = rgb(230, 230, 230)  # 90%
grey2 = rgb(204, 204, 204)  # 80%
grey3 = rgb(179, 179, 179)  # 70%
grey4 = rgb(153, 153, 153)  # 60%
grey5 = rgb(128, 128, 128)  # 50%
grey6 = rgb(102, 102, 102)  # 40%
grey7 = rgb(77, 77, 77)     # 30%
grey8 = rgb(51, 51, 51)     # 20%
grey9 = rgb(26, 26, 26)    # 10%, very dark

# DVD-specific colors from the legend
# bright_green, teal_green, deep_blue
bright_green = rgb(174, 233, 72)   # Bright green (DVD-S)
teal_green = rgb(49, 162, 142)    # Teal green (DVD-B)
deep_blue = rgb(54, 78, 128)     # Deep muted blue (DVD-P)