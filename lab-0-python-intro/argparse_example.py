
import os

if __name__ == '__main__':
    # imports can occur any where in the program
    # it is good practice to put module dependencies at the top of the file
    # and CLI dependency imports inside the `__name__ == '__main__'` conditional block
    # so they are only imported when the script is run as a program and not imported as a library 
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN on MNIST dataset')
    parser.add_argument('dataset-dir', 
                        type=str, 
                        help='Directory in which to download')
    parser.add_argument('--stride', 
                        type=str, 
                        default='2x2')
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=32)

    args = parser.parse_args()
    print(args)