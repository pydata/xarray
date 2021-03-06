def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "attach_marks(marks): function to attach marks to tests and test variants",
    )


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        mark = item.get_closest_marker("attach_marks")
        if mark is None:
            continue
        index = item.own_markers.index(mark)
        del item.own_markers[index]

        marks = mark.args[0]
        if not isinstance(marks, dict):
            continue

        variant = item.name[len(item.originalname) :]
        to_attach = marks.get(variant)
        if to_attach is None:
            continue

        for mark in to_attach:
            item.add_marker(mark)
