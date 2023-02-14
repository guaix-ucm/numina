
import argparse
import os.path

from numina.array.display.fileinfo import FileInfo


def subsets_of_fileinfo_from_txt(filename):
    """Returns a dictionary with subsets of FileInfo instances from a TXT file.

    Each subset of files must be preceded by a line:
    @ <number> <label>
    where <number> indicates the number of files in that subset,
    and <label> is a label for that subset. Any additional text
    beyond <label> in the same line is ignored.

    Note that blank lines or lines starting by the hash symbol are
    also ignored. The name of the files comprising each subset will be
    obtained from the first contiguous character string in every
    line (thus, the rest of the line will be discarded).
    
    Parameters
    ----------
    filename : string
        Name of a TXT file containing a list of FITS files grouped
        in different subsets by the @ symbol.

    Returns
    -------
    dict_of_subsets_of_fileinfo : dictionary
        Dictionary containing as many entries as different subsets
        of files available. Each value of the dictionary is a
        dictionary with a label (sequential number starting at zero)
        and the list of FileInfo instances within subset.
        
    """
    
    # check for input file
    if not os.path.isfile(filename):
        raise ValueError("File " + filename + " not found!")

    # read input file
    with open(filename) as f:
        file_content = f.read().splitlines()

    # obtain the different subsets of files
    dict_of_subsets_of_fileinfo = {}
    label = None
    sublist_of_fileinfo = []
    idict = 0
    ifiles = 0
    nfiles = 0
    sublist_finished = True
    for line in file_content:
        if len(line) > 0:
            if line[0] != '#':
                if label is None:
                    if line[0] == "@":
                        nfiles = int(line[1:].split()[0])
                        label = line[1:].split()[1]
                        sublist_of_fileinfo = []
                        ifiles = 0
                        sublist_finished = False
                    else:
                        raise ValueError("Expected @ symbol not found!")
                else:
                    if line[0] == "@":
                        raise ValueError("Unexpected @ symbol found!")
                    tmplist = line.split()
                    tmpfile = tmplist[0]
                    if len(tmplist) > 1:
                        tmpinfo = tmplist[1:]
                    else:
                        tmpinfo = None
                    if not os.path.isfile(tmpfile):
                        raise ValueError("File " + tmpfile + " not found!")
                    sublist_of_fileinfo.append(FileInfo(tmpfile, tmpinfo))
                    ifiles += 1
                    if ifiles == nfiles:
                        dict_of_subsets_of_fileinfo[idict] = {}
                        tmpdict = dict_of_subsets_of_fileinfo[idict]
                        tmpdict['label'] = label
                        tmpdict['list_of_fileinfo'] = sublist_of_fileinfo
                        idict += 1
                        label = None
                        sublist_of_fileinfo = []
                        ifiles = 0
                        sublist_finished = True

    if not sublist_finished:
        raise ValueError("Unexpected end of sublist of files.")

    return dict_of_subsets_of_fileinfo


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_file",
                        help="txt file with list of subsets of images")
    args = parser.parse_args(args)

    # execute function
    dict_of_subsets = subsets_of_fileinfo_from_txt(args.txt_file)
    for idict in range(len(dict_of_subsets)):
        tmpdict = dict_of_subsets[idict]
        print('\n>>> Label: ', tmpdict['label'])
        print('>>> List of FileInfo instances:')
        for item in tmpdict['list_of_fileinfo']:
            print(item)


if __name__ == "__main__":

    main()
