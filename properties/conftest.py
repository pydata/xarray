from hypothesis import settings

# Run for a while - arrays are a bigger search space than usual
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")
