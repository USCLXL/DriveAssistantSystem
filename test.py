import math

label2label = {"ambulance": 'car', 'auto': 'car', 'bicycle': 'bike', 'garbagevan': 'car', 'human': 'car',
               'minibus': 'car', 'minivan': 'car', 'motorbike': 'motor', 'Pickup': 'car',
               'army': 'car', 'policecar': 'car', 'rickshaw': 'motor', 'scooter': 'motor', 'suv': 'car',
               'three': 'motor', 'van': 'car', 'wheelbarrow': 'motor'}


def adapt_label(label):
    true_label = label.split(' ')[0]
    if label in label2label:
        print(label2label[true_label])


if __name__ == '__main__':
    from string import digits
    label = 'car 0.6'
    a = 3

    remove_digits = str.maketrans('', '', digits + '.')
    real_label = label.translate(remove_digits).rstrip()
    print(real_label)
