

def experiment_loglikely(datasets_location="numerai_datasets/October/", algorithm="kerasAlgorithms.MLP"):
    try:
         model = __import__(algorithm)
    except ImportError as e:
        print(e)
    else:
        try:
            print(model.loglikely(datasets_location))
            print(algorithm)
        except ImportError as e:
            print("Import error({0}): {1}".format(e.errno, e.strerror))
    
if __name__ == "__main__":
    import sys
    try:
        if (sys.argv[1] == "-h"):
            print("try: python numerai.py experiment")
        if (sys.argv[1] == "experiment"):
            try:
                datasets_location = sys.argv[2]
                algorithm = sys.argv[3]
                experiment_loglikely(datasets_location, algorithm)
            except (ValueError):
                print("suggested usage: python numerai.py experiment datasets_location algorithm")
            except (IndexError):
                print("running default experiment")
                experiment_loglikely()
        else:
            print("try: 'python  experiment' to run an experiment")
    except(IndexError):
        print("python numerai.py -h for usage")
