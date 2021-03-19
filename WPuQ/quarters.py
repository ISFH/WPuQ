quarter_names = {'Huegelshart': '4',
            'Ohrberg': '2'}

heat_objs = {'Huegelshart': [],
             'Ohrberg': [13, 19, 22]}

pv_objs = {'Huegelshart': [],
           'Ohrberg': {
               13: 2.5,
               15: 4.5,
               #19: 6.5, does not use its own PV electricity
               26: 0.75,
               #30: 4.5, does not use its own PV electricity
               33: 4.0,
               49: 4.0}
           }