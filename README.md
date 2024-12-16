# Explainable Transformer-Based Classification and Segmentation of Intracranial Hemorrhages
Team members: 

- Rupali Sinha | rsinha@bu.edu
- Albert Zhao | albertz@bu.edu
- Ishan Bhattacharjee | ibhattac@bu.edu
- Benjamin Axline | baxline@bu.edu

[📄 View Proposal (PDF)](ACTUAL523%20Project%20Proposal%20Template.pdf)


Abstract: 
We are creating a fine-tuned segmentation model for classifying Intracranial Hemorrhage CT Detection. 
We are running a segmentation model as a base ground-truth, and then subsequently running a 
vision transformer that is fine tuned with explainability methods. We will then compare these two methods 
on our data, to determine which model is more accurate for classifying the subtype of hemorrhage. 


Data Installation: 
Run this line of code to install dataset. 
kaggle competitions download -c rsna-intracranial-hemorrhage-detection


