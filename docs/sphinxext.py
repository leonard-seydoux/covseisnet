from sphinx_gallery.scrapers import matplotlib_scraper


def matplotlib_svg_scraper(*args, **kwargs):
    return matplotlib_scraper(*args, format="png", **kwargs)


def reset_mpl(gallery_conf, fname, when):
    import matplotlib as mpl

    grey = "0.4"

    mpl.rcParams["lines.linewidth"] = 0.7
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["axes.facecolor"] = (1, 1, 1, 0.07)
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.edgecolor"] = grey
    mpl.rcParams["axes.labelcolor"] = grey
    mpl.rcParams["text.color"] = grey
    mpl.rcParams["xtick.color"] = grey
    mpl.rcParams["ytick.color"] = grey
    mpl.rcParams["axes.titlecolor"] = grey
    mpl.rcParams["grid.color"] = grey
