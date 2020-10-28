
import sys
import csv


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inp = sys.argv[1]
    features = sys.argv[2]

    tmp_names = []
    tmp_xyz = []
    # read CSV
    with open(features, newline='') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            print(', '.join(row))
            if i > 2:
                tmp_xyz.append(row[1:4])
                tmp_names.append(row[11])
            i = i + 1

    outp = sys.argv[3]


    nrrd_in = nrrd.read(nrrd_file)




    nif = nib.load(nifty_inp)
    nif2 = nib.Nifti1Image(nrrd_in[0], np.eye(4))
    nib.save(nif2,outp)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout

