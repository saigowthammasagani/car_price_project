from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=180).run()
assert not at.exception, at.exception
print("tabs:", len(at.tabs), "| metrics:", [m.value for m in at.metric][:4])
at.button[0].click().run()
assert not at.exception, at.exception
print("prediction:", at.metric[-1].value)
