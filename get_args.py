
def get_args():
    # Description
    parser = argparse.ArgumentParser(description='Train the Res_Net on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cuda', type=str, help='Visible cuda device', default='0')
    
    # Training
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=3,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=7,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0000001,
                        help='Learning rate', dest='lr')
    
    # Model
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    
    return parser.parse_args()
