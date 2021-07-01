# Define functions
def label_converter(label):
    if label == 'ham':
        return 0
    elif label == 'spam':
        return 1
    else:
        return 'Error'