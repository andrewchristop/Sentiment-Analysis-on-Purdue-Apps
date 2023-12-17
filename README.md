# Sentiment Analysis using LSTM on Purdue-based Applications

As Purdue’s student body grows year after year, the university felt it necessary to build applications tied to certain campus services to provide students and faculty better experience. We have managed to find 5 Purdue published applications from the Apple App Store at the time of doing this project. They are:

* Purdue Guide 
* Purdue App
* Purdue Athletics
* Purdue RecWell
* Purdue Mobile Menus

Given that Purdue had to maintain all 5 applications, and reading the somewhat mixed reviews from the AppStore, we suspected that Purdue’s resources (both human and technical) may be spread too thin. Now it is entirely possible that Purdue may have made leaps and bounds in improving user experience with these applications, however, the reviews were rather outdated making it impossible for us to confirm this fact. As such, we decided to gauge user responses on these applications by using Google Forms to gather the most recent feedback on the performance of these applications. Our use of sentiment analysis to categorize reviews as either positive or negative in tone hopes to acknowledge the significant role that digital applications and services play in contemporary student life. We hope that our findings can provide Purdue affiliated developers with actionable insights on the various components of the app that should be kept in response to the positive reviews and ones that needed to be done away with or improved with regards to user reviews that are negative in tone. 

Models are already pre-compiled and saved in the HDF5 (.h5) format so you won't have to run all 3 epochs which will take approximately 45 minutes to run altogether. Training and testing accuracy are in the high 90%. Last known estimates put the train-test accuracy around 96-97%. Below is an excerpt gathered from the program output taken from the LSTM model predicting actual data from the surveys that we gathered.

```
App: Purdue RecWell
Review: I do not use any of the Purdue affiliated apps that much.
Predicted Sentiment: Negative

App: Purdue RecWell
Review: Very Slow and Laggy
Predicted Sentiment: Negative

App: Purdue Guide
Review: Perfect
Predicted Sentiment: Positive
```

But there are times when the model does mess up.

```
App: Purdue App
Review: It's an okay app but I rarely use it. It's been sitting on my phone for a year which reminds me that I should delete it because it's just taking up space
Predicted Sentiment: Positive

App: Purdue Athletics
Review: Can't login. My email password did not work. Requested new password. The temp password did not work to reset my access.   Not sure what the issue is here. Last app was better
Predicted Sentiment: Positive
```

My hypothesis on this would be the fact that there wasn't enough training data to train and test the model on. In addition, the reviews of the training data were poor in quality. Almost every review was littered with spelling, grammatical errors, or a combination of the two. I have however yet to test this hypothesis out in the near future.


Please feel free to introduce your own set of data to have the model predict a sentiment for you. There are two ways you can go about this. You can either make changes to main.py or you can modify your .csv file to match the column headers specified in main.py. Good luck either way.  



To install dependencies please execute `pip install -r requirements.txt` from the project directory


To run the code please execute `python3 main.py`


**DEPENDENCIES LISTED in requirements.txt may not be exhaustive. Please follow instructions provided by the interpreter to resolve issues with dependencies**
