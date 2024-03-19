import matplotlib.pyplot as plt
from IPython import display

# Turn on interactive mode for plotting
plt.ion()


# Function to plot scores and mean scores
def plot(scores, mean_scores):
    # Clear the current figure and display
    display.clear_output(wait=True)
    display.display(plt.gcf())

    # Clear the current plot
    plt.clf()

    # Set plot title and labels
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot scores and mean scores
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')

    # Set y-axis lower limit to 0
    plt.ylim(ymin=0)

    # Display the current score and mean score as text on the plot
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    # Show the plot
    plt.show(block=False)

    # Pause for a short duration to allow the plot to update
    plt.pause(.1)
