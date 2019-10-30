try:
    from hypothesis import settings
except ImportError:
    pass
else:
    # Run for a while - arrays are a bigger search space than usual
    settings.register_profile("ci", deadline=None, print_blob=True)
    settings.load_profile("ci")
