from spp.plotting import (
    create_enu_figure,
    create_quality_figure,
    create_skyplot_figure,
    save_figure,
)


def test_create_figures(short_result, tmp_path):
    for factory, name in (
        (create_enu_figure, "enu"),
        (create_quality_figure, "quality"),
        (create_skyplot_figure, "sky"),
    ):
        figure = factory(short_result)
        assert figure is not None
        path = save_figure(figure, tmp_path / f"{name}.png", dpi=80)
        assert path.is_file()
        assert path.stat().st_size > 1000
