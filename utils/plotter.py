import matplotlib.pyplot as plt


def plotter(results, output_filename):
    # plot regrets and save results
    plt.plot(results)
    plt.xlabel('Time')
    plt.ylabel('Cumulative reward')
    plt.show()

    with open(output_filename, "w") as outfile:
        for num in results:
            outfile.write('{0}'.format(num))
            outfile.write("\n")
