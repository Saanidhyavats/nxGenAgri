from disease_description import disease_dic

for elem in disease_dic:
    elem = """<div style="background-color: lightcoral;>""" + elem
    print(elem)
    