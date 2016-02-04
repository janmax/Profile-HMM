def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(mesg="Elapsed time is "):
    import time
    if 'startTime_for_tictoc' in globals():
        # print (mesg + "{0} seconds.".format(time.time() - startTime_for_tictoc))
        return time.time() - startTime_for_tictoc
    else:
        print ("Toc: start time not set")
