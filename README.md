# Twitter Analysis
There are 3 problem handled by this solution:  

1> find the trending topics about celebrity
	a. Input:  list of celebrity name and number of topics (how many trending topics we want)
     command line argument (example): python content_analysis.py -t topics -c "bradley cooper, clint eastwood, chris kyle" -n 20
	b. Output: json file (result.json) having celebrity name and related topics

2> find the celebrity name on the basis of new tweet
	a. Input: any sentence (new tweet)
	command line argument (example): python content_analysis.py -t prediction -s "AmericanSniper Chris Kyle's widow This movie will be how 	my kids remember their dad“
	b.  Output: Celebrity name (based on best match) on command prompt 

3> break the twitter hashtags into proper words (Note: it is taking long execution time)
	a. Input for command line argument: python content_analysis.py -t tag
	b. Output: /hashtag_keyword.json
