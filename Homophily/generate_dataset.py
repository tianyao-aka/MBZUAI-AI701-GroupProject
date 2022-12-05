import psutil
from utils import *
import argparse
import schedule

if __name__=='__main__':
    import time
    # s = time.time()
    index = 0
    parser = argparse.ArgumentParser(description='generating dataset')
    parser.add_argument('--dset_name', type=str, default='Cora',
                        help='name of the dataset')
    args = parser.parse_args()
    name = args.dset_name
    def process_node_image():
        global index
        ram = psutil.virtual_memory()[2]
        if ram>65:
            print ('current ram used:',ram)
            return
        if index>100:
            schedule.cancel_job()
            exit()
        else:
            os.system(f'python utils.py --index {index} --dset_name {name} & ')
            print (f'running utils to generate node image for index:{index},current ram usage is:{ram}')
            index +=1

    job = schedule.every(30).seconds.do(process_node_image)

    while True:
        schedule.run_pending()


