#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

def overplot_ds9reg(filename, ax):
    """Overplot a ds9 region file.

    Parameters
    ----------
    filename : str
        File name of the ds9 region file.
    ax : matplotlib axes instance
        Matplotlib axes instance.

    """

    # read ds9 region file
    with open(filename) as f:
        file_content = f.read().splitlines()

    # check first line
    first_line = file_content[0]
    if "# Region file format: DS9" not in first_line:
        raise ValueError("Unrecognized ds9 region file format")

    for line in file_content:
        if line[0:4] == "line":
            line_fields = line.split()
            x1 = float(line_fields[1])
            y1 = float(line_fields[2])
            x2 = float(line_fields[3])
            y2 = float(line_fields[4])
            if "color" in line:
                i = line.find("color=")
                color = line[i+6:i+13]
            else:
                color = "green"
            ax.plot([x1,x2], [y1,y2], '-', color=color)
        elif line[0:4] == "text":
            line_fields = line.split()
            x0 = float(line_fields[1])
            y0 = float(line_fields[2])
            text=line_fields[3][1:-1]
            if "color" in line:
                i = line.find("color=")
                color = line[i+6:i+13]
            else:
                color = "green"
            ax.text(x0, y0, text, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="grey", ),
                    color=color, fontweight='bold', backgroundcolor='white',
                    ha='center')
        else:
            # ignore
            pass
