# Procedure Coder

[The Insight demo web app](http://procedurecoder.site/input)

This project was done as a consulting project for [drChrono](https://www.drchrono.com/)

Current procedural terminology [(CPT)](https://en.wikipedia.org/wiki/Current_Procedural_Terminology) codes are insurance billing codes used by providers for the purposes of reimbursement. These codes are assigned to a doctor's note by a dedicated biller. In order to reduce the burden on billers, increase the code assignment accuracy, and simplify the code assignment for doctors, I have developed an app to recommend appropriate CPT codes --- Procedure Coder.

The Procedure Coder app provides ranked recommended CPT codes for a given body of text. It uses a set of multinomial Naive Bayes (NB) classifiers in conjunction with cosine similarity to identify the most applicable CPT codes. Each NB classifier is trained to specifically identify notes belonging to a particular group of CPT codes (such as Outpatient care). The cosine similarity recovers the closest CPT code descriptions to the text, which in essence identifies important keywords between the text and the description.
