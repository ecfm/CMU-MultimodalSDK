import argparse
import urllib2
import os.path
from subprocess import call
import sys

parser = argparse.ArgumentParser(description="Specify the datasets you want to download.")
parser.add_argument("--dataset", help="Specify the name of the dataset you want, in uppercase letters.", type=str)

args = parser.parse_args()

datasets = {
    "MOSI": "http://sorena.multicomp.cs.cmu.edu/downloads/MOSI.tar.gz",
    "POM": "url.for.pom.tar",
    "IEMOCAP": "url.for.iemocap.tar"
}

if args.dataset == None:
    print "\nTry python downloader.py -h for usage information\n"
elif args.dataset not in datasets.keys():
    print "\nThe dataset you specified is not provided! Please check the available datasets:\n"
    for dataset in datasets.keys():
        print dataset + "\n"
else:
    url = datasets[args.dataset]
    file_name = url.split('/')[-1]
    file_path = os.path.join('temp', file_name) # temp path for storing the zip file
    output_path = os.path.join("..", "datasets") # path for the folder storing the directory
    target = os.path.join("..", "datasets", file_name.split(".")[0]) # path of the dataset directory
    call(['mkdir', output_path]) # if datasets directory not present mv command will have different behavior later
    
    giveup_or_down = None
    if os.path.exists(target):
        while giveup_or_down not in ['Y', 'N']:
            giveup_or_down = raw_input("Dataset folder already exists, do you want to override it? [Y]es\\[N]o\n")
            giveup_or_down = giveup_or_down.upper()
        if giveup_or_down == 'N':
            sys.exit()
        elif giveup_or_down == 'Y':
            call(['rm', '-r', target])

    extract_or_down = None
    if os.path.exists(file_path):
        while extract_or_down not in ['E', 'D']:
            extract_or_down = raw_input("Zip file already exists in temp, do you want to extract it or re-download? [E]xtract\\[D]ownload\n")
            extract_or_down = extract_or_down.upper()
    if extract_or_down == None or extract_or_down == 'D':
        call(['mkdir', '-p', 'temp'])

        file_name = url.split('/')[-1]
        file_path = os.path.join('temp', file_name)

        u = urllib2.urlopen(url)
        with open(file_path, 'wb') as f:
            meta = u.info()
            file_size = int(meta.getheaders("Content-Length")[0])
            print "Downloading: {}, size: {}".format(file_name, file_size)

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                sys.stdout.write('\r')
                # status = r"%10d bytes [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                sys.stdout.write("[%-20s] [%3.2f%%]" % ('='*int((file_size_dl * 100. / file_size)/5), file_size_dl * 100. / file_size))
                # status = status + chr(8) * (len(status) + 1)
                # sys.stdout.write(status)
                sys.stdout.flush()


    print "\nExtracting dataset...\n"
    call(["tar", "-xzf", file_path])
    call(['mv', file_name.split(".")[0], output_path])
    call(['rm', '-r', 'temp'])
    if os.path.exists(file_name.split(".")[0]):
        call(['rm', '-r', file_name.split(".")[0]])
    print "{} dataset should be good to go. Refer to the README.md on how to load the data.".format(file_name.split(".")[0])