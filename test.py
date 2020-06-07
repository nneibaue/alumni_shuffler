from zoomSesh import ZoomSesh

a = ZoomSesh(max_people=10)

alumni = a.alumni

b = alumni[alumni.index.isin([2,3,4])]

print(b)


