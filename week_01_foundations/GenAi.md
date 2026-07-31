## Week 1: Orientation: AI, ML, Deep Learning, GenAI

### DAY 1

- AI -> Artificial Intelligence
- ML -> Machine Learning
- DL -> Deep Learning
- GenAI -> Generative AI
- LLM -> Large Language Model

#### AI: AI is a computer technology that utilizes human intelligence to make decisions, learn, and act. 

#### ML: ML is a machine that learns from data and give prediction.

#### DL: DL is a type of machine learning which based on neural network that powers advanced technologies like voice recognition, image processing etc.

#### GenAI: GenAI is a type of AI, which learn from large data and make new content(Generating text, image etc).

#### LLM: LLM is a type of GenAI, which learn from large data and generate text in response. Now many LLM can generate image, video also.

*All LLM is GenAI but all GenAI is not LLM*

-----

### DAY 2

#### Classical ML vs GenAI

| Aspect | Classical ML | GenAI (LLMs) |
|---|---|---|
| **Output type** | Predicts a fixed label/category | Generates free-form content (text, images, etc.) |
| **Explainability** | Usually a black box (needs SHAP/LIME for explanations) | Can natively explain why it made a decision in plain language |
| **Training data need** | Needs large labeled dataset (thousands of headlines tagged by class) | Works with few-shot or even zero-shot examples; no large labeled dataset required |
| **Feature engineering** | Requires manual feature extraction (TF-IDF, n-grams, word counts, punctuation patterns, etc.) | No manual feature engineering; understands raw text directly |
| **Task scope** | Narrow — solves only the task it was trained for (classification only) | Broad — same model can classify, summarize, rewrite, translate, generate |
| **Latency/Cost** | Fast, cheap inference once trained (lightweight models like Logistic Regression, SVM) | Slower and costlier per request (large transformer models) |
| **Multimodal capability** | Typically single modality (usually just text or numeric features) | Can handle text, images, audio together (e.g., classify a clickbait thumbnail + headline combo) |
| **Consistency** | Very consistent — same input always gives same output | Can vary slightly between runs (unless temperature=0), less deterministic |



**Quick clickbait example end-to-end:**

-   **Input:** _"You Won't Believe What This Dog Did Next!"_
-   **Classical ML (e.g., trained SVM/Logistic Regression on TF-IDF features):**  
    → Output: `Label: Clickbait`
-   **GenAI (e.g., LLM prompted for classification):**  
    → Output:
    `
 Label: Clickbait`
 `Reason: Uses curiosity-gap phrasing ("You Won't Believe") and vague subject ("What...Next") to withhold information and provoke clicks.`
`Suggested rewrite: "Dog Rescues Owner From House Fire, Video Shows"
`

----

### DAY 3

### Discriminative vs Generative Models

#### Discriminative models

Discriminative models learn to predict a specific output (y) from an input (x):

$$
P(y \mid x)
$$

They usually select an answer from predefined labels.

**Example 1: Clickbait classification**

-   Input: A Bengali social-media post
    
-   Output: Label 0, 1, 2, or 3
    

**Example 2: Email spam detection**

-   Input: An email
    
-   Output: Spam or not spam
    

### Generative models

Generative language models learn patterns in token sequences and generate new content, usually one token at a time:

$$ 
P(x_t \mid x_1,\ldots,x_{t-1})  
$$

**Example 1: Headline rewriting**

-   Input: A clickbait headline
    
-   Output: A newly generated neutral headline
    

**Example 2: Article summarization**

-   Input: A long article
    
-   Output: A newly generated summary
    

### Where does XLM-R belong?

XLM-R is an **encoder-only pretrained Transformer**. In your clickbait project, it is fine-tuned to predict one of four predefined labels.

Therefore, it belongs to:

> **Discriminative deep learning or Transformer-based classification**

It is neither classical ML nor a generative LLM.

```text
Input text
   ↓
XLM-R encoder
   ↓
Classification layer
   ↓
Label 0, 1, 2, or 3

```

----

### DAY 4

Perform **two small experiments**—one using only **text** and another using both **an image and text**—and document their weaknesses.

### Task 1: Text-based experiment


**Tool/model used:**  ChatGPT
**Input:**  আপনি বিশ্বাসই করতে পারবেন না এরপর কী ঘটল!
**Generated output:**  
**Predicted label: 3 — Strong clickbait**
**Explanation:** The headline deliberately withholds the actual event and uses an exaggerated curiosity gap—“আপনি বিশ্বাসই করতে পারবেন না” (“You won’t believe”)—to pressure readers into clicking.
**Neutral rewrite:** “এরপর যা ঘটেছিল, তার বিস্তারিত জানুন।”

**Limitations observed:**
- Yes, the classification was reasonable.
- Yes, the model did follow the label definitions.
- No, it did not add any unsupported information.
- Yes, some changes in output rewrite section, when the same prompt was repeated. There is completely different rewrite section. (“এরপর ঘটে যাওয়া ঘটনার বিস্তারিত প্রকাশ করা হয়েছে।”)
- Yes, it did understand the Bengali text properly.



**Tool/model used:** Claude
**Input:** আপনি বিশ্বাসই করতে পারবেন না এরপর কী ঘটল!
**Generated output:** 
**Predicted Label: 3 (Strong clickbait)**

**Explanation:** The headline "আপনি বিশ্বাসই করতে পারবেন না এরপর কী ঘটল!" (translation: "You won't even believe what happened next!") is a textbook example of strong clickbait. It uses:

-   A direct emotional/curiosity-provoking appeal to the reader ("আপনি বিশ্বাসই করতে পারবেন না" – "you won't be able to believe")
-   Complete withholding of the actual subject or event
-   Vague, sensational phrasing ("এরপর কী ঘটল" – "what happened next") with no context about who, what, or where
-   An exclamation mark reinforcing hype/urgency

**Neutral rewrite:** "এই ঘটনায় যা ঘটেছিল, বিস্তারিত জানুন" ("Learn the details of what happened in this event"),
Although this is still vague, since the original provides no real information to neutralize it.

**Limitations observed:**
- Yes, the classification was reasonable.
- Yes, the model did follow the label definitions.
- No, it did not add any unsupported information.
- No, the output did not change when the same prompt was repeated.
- Yes, it did understand the Bengali text properly.



**Tool/model used:**  Gemini
**Input:**  আপনি বিশ্বাসই করতে পারবেন না এরপর কী ঘটল!
**Generated output:**  
**Predicted Label:** **Label 3: Strong clickbait**

**Explanation:** The headline employs extreme sensationalism ("আপনি বিশ্বাসই করতে পারবেন না" / _You won't believe_) and deliberately withhold key information ("এরপর কী ঘটল!" / _what happened next!_). This classic "curiosity gap" manipulation creates an intense urge to click without providing any actual context or substance about the event.

**Neutral Rewrite:**  **"ঘটনাটির পরবর্তী আপডেট"** (Subsequent updates on the event)
(Note: A precise rewrite requires knowing the actual story, e.g., "দুর্ঘটনার পর উদ্ধারকাজ শুরু করল পুলিশ" / Police began rescue operations after the accident).

**Limitations observed:**
- Yes, the classification was reasonable.
- Yes, the model did follow the label definitions.
- No, it did not add any unsupported information.
- Yes, some changes in output rewrite section, when the same prompt was repeated. There are different rewrite section. ("ঘটনাটির পরবর্তী বিস্তারিত তথ্য")
- Yes, it did understand the Bengali text properly.


Among the three models, **Claude provided the most precise and explainable answer**. It also produced the same response when the prompt was repeated, demonstrating strong consistency. The other two models provided less clear and less effective explanations than **Claude**.


### Task 2: Test a multimodal use case


**Tool/model used:** ChatGPT
**Uploaded image:** Ney-vini-Raph.png
**Generated output:**  
1.  **What the image shows:** Three football players stand together on a stadium pitch, wearing Brazil’s yellow-and-blue national-team uniforms. The player in the middle is smiling, while the others make celebratory hand gestures.
    
2.  **Most important observation:** The image highlights the three Brazilian players posing together as teammates. It does not display a match result or score.
    
3.  **Visible numbers and labels:**
    
    -   Jersey numbers: **10**, **7**, and **11**
        
    -   **BRASIL**
        
    -   **CBF** on the Brazilian football crest
        
    -   Nike swoosh logos
        
4.  **Simple explanation:** This appears to be a posed or celebratory photograph of three Brazil national-team players at a football stadium.
    
5.  **Evidence limitation:** No scoreboard, opponent, match date, competition name, or final score is clearly visible, so these details cannot be determined from the image alone.

**Limitations observed:**
- Yes, it did read all numbers correctly.
- Yes, it did understand the labels and axes.
- Yes, it overlooked an important information and that is the players name. It can't identify the exact players.
- No, it did not make unsupported claims.
- No, image resolution did not affect the response.



**Tool/model used:** Claude
**Uploaded image:** Ney-vini-Raph.png
**Generated output:**  

**1. Description:** The image shows three men wearing Brazil national football team jerseys (yellow with green trim, blue shorts), standing arm-in-arm on a stadium pitch with a crowd and stadium lights blurred in the background. They are posing for a photo, with the man on the left making a "hang loose" hand gesture and the man on the right making a fist gesture.

**2. Most important "result":** This appears to be a posed team photograph/celebration shot rather than an image depicting a specific match result, score, or event outcome. There is no scoreboard, match context, or other evidence of a game result visible in the image itself.

**3. Visible numbers/labels:**

-   Jersey number **10** (left player, also printed on his shorts)
-   Jersey number **7** (center player, also printed on his shorts)
-   Jersey number **11** (right player, also printed on his shorts)
-   "BRASIL" text on all three jerseys
-   A CBF (Brazilian Football Confederation) crest with stars on each jersey
-   Nike logo (swoosh) on each jersey
-   A partial patch/badge on the upper right sleeve of the right player (text not clearly legible)

**4. Simple explanation:** The picture shows three teammates on Brazil's national football squad, identified by their shirt numbers (10, 7, and 11), posing together on the field, seemingly in a celebratory or promotional moment.

**5. Note on limitations:** I can't independently verify the identities of the individuals shown, confirm this is an actual, unaltered photograph from a real match, or determine the specific date, opponent, or event context — none of that information is present in the image itself. I'd note that the uploaded file name suggests this may be an AI-generated image rather than an authentic photograph, so I can't confirm it depicts a real event at all.

**Limitations observed:**
- Yes, it did read all numbers correctly.
- Yes, it did understand the labels and axes.
- Yes, it overlooked an important information and that is the players name. It can't identify the exact players.
- No, it did not make unsupported claims.
- No, image resolution did not affect the response.



**Tool/model used:** Gemini
**Uploaded image:** Ney-vini-Raph.png
**Generated output:**  
Here is an analysis based strictly on the content of the image provided:

**1. Description of the Image**

The image shows three male professional football (soccer) players standing side by side with their arms around each other's shoulders inside a stadium. They are wearing matching national team uniforms consisting of yellow short-sleeved jerseys with green trim, blue shorts, and green Nike swoosh logos.

-   **Left:** Neymar Jr., wearing number 10, making a hand gesture ("shaka" sign).
    
-   **Middle:** Vinícius Júnior, wearing number 7, smiling towards the camera.
    
-   **Right:** Raphinha, wearing number 11, posing with a clenched fist.
    

**2. Most Important Result**

The key visual subject of the image is the **unity and camaraderie of the Brazilian national football team's attacking line (Neymar, Vinícius Jr., and Raphinha)** posing together on the pitch in their official kit.

**3. Visible Numbers and Labels**

#### Numbers

-   **Jersey & Shorts #10:** Worn by the player on the left (Neymar Jr.).
    
-   **Jersey & Shorts #7:** Worn by the player in the middle (Vinícius Jr.).
    
-   **Jersey & Shorts #11:** Worn by the player on the right (Raphinha).
    

#### Crests and Text Labels

-   **CBF Crest / "BRASIL":** Printed on the chest of each yellow jersey and on the lower leg of the blue shorts.
    
-   **Nike Logos:** Green Nike swoosh visible on the right chest of each jersey and on the left side of each pair of shorts.
    
-   **FIFA / Tournament Patch:** A patch visible on the left sleeve of the player wearing number 11.
    

**4. Explanation in Simple Language**

This photo shows three star players from the Brazil national football team standing together as teammates. By posing closely with their arms around one another, wearing their official team jerseys (#10, #7, and #11), they are displaying team spirit and solidarity during a football match or event.

**Limitations observed:**
- Yes, it did read all numbers correctly.
- Yes, it did understand the labels and axes.
- No, it did not overlook any important information but there is no limitation as it seems everything is true that can be tricky.
- No, it did not make unsupported claims.
- No, image resolution did not affect the response.


In this task, **Gemini identified all three players**, whereas **Claude and GPT could not identify them confidently**. Claude and GPT included a **limitations section** explaining why the players could not be reliably identified. In contrast, Gemini did not acknowledge any uncertainty or limitations and presented its identifications as factual. If those identifications are incorrect or unsupported, this could mislead the reader.

----

### DAY 5

####  GenAI Risk Register

A **risk register** is a structured list of possible problems, their effects, and the actions used to reduce them.
| No. | Risk | Description and Possible Impact | Mitigation |
|---:|---|---|---|
| 1 | **Hallucination** | The model may generate convincing but factually incorrect information. | Verify important claims using reliable sources. |
| 2 | **Bias and unfairness** | The model may reproduce social, cultural, gender, linguistic, or political biases from its training data. | Use diverse evaluation data. |
| 3 | **Privacy and data leakage** | Users may submit private, personal, medical, academic, or organizational information that could be exposed or processed improperly. | Remove identifying information; avoid uploading confidential data without authorization. |
| 4 | **Copyright and intellectual property** | Users may upload copyrighted data without permission. | Use properly licensed data. |
| 5 | **Overreliance** | Users may accept AI output without checking it. | Clearly communicate limitations; require human approval for high-impact decisions. |

-------


## DAY 6

**The project will eventually have this structure:**

genai-learning/
├── notebooks/
│   └── 01_environment_test.ipynb
├── src/
├── data/
├── README.md
├── requirements.txt
└── .gitignore

###  GitHub Repository

[View the GenAI Project on GitHub](https://github.com/Mansib-eng/GenAI-Project)

----


## DAY 7


### Week 1: Generative AI Quiz

**Total questions:** 15  
**Total marks:** 15  
**Instructions:** Answer the questions without checking your notes. Each question carries 1 mark.


### Multiple-Choice Questions

**1. Which hierarchy is the most accurate?**

- A. AI → GenAI → Machine Learning → LLM
- B. AI → Machine Learning → Deep Learning → GenAI → LLM
- C. Machine Learning → AI → LLM → GenAI
- D. LLM → GenAI → Deep Learning → AI

**Answer:** B


**2. What is the main purpose of a classical classification model?**

- A. Generate new images
- B. Predict a predefined label from an input
- C. Write a new article
- D. Produce speech from text

**Answer:** B


**3. Which task is the clearest example of Generative AI?**

- A. Predicting whether an email is spam
- B. Predicting a customer’s credit-risk score
- C. Generating a neutral rewrite of a clickbait headline
- D. Classifying a tumour as benign or malignant

**Answer:** C


**4. What does a discriminative model generally learn?**

- A. $P(y \mid x)$
- B. $P(x)$ only
- C. How to generate images
- D. How to store documents

**Answer:** A


**5. A decoder-based generative language model primarily generates text by:**

- A. Selecting a predefined class
- B. Predicting the next token based on previous tokens
- C. Searching a database for every word
- D. Copying complete responses from its training data

**Answer:** B


**6. Where does a fine-tuned XLM-R clickbait classifier belong?**

- A. Classical machine learning
- B. Image-generative AI
- C. Discriminative Transformer-based deep learning
- D. Decoder-based generative AI

**Answer:** C


**7. Which one is a multimodal GenAI task?**

- A. Using only numerical features to predict house prices
- B. Uploading a graph and asking the model to explain it
- C. Using Logistic Regression to classify text
- D. Sorting a spreadsheet alphabetically

**Answer:** B


**8. What is hallucination in Generative AI?**

- A. The model takes too long to answer
- B. The model generates convincing but incorrect information
- C. The model refuses to produce an image
- D. The model uses too much memory

**Answer:** B


**9. Which action best protects privacy when using an external GenAI service?**

- A. Uploading all available data for better results
- B. Including passwords so the model can access resources
- C. Removing identifying and confidential information before submission
- D. Asking the model not to remember the information

**Answer:** C


**10. Why do we use a virtual environment such as `.venv`?**

- A. To increase internet speed
- B. To keep project packages isolated from other Python projects
- C. To upload the project automatically to GitHub
- D. To replace the operating system

**Answer:** B


**11. Which command activates `.venv` in Windows PowerShell?**

- A. `python activate .venv`
- B. `git activate .venv`
- C. `.venv\Scripts\Activate.ps1`
- D. `pip activate environment`

**Answer:** C


**12. What is the purpose of `.gitignore`?**

- A. It removes Git from the project
- B. It tells Git which files and folders should not be tracked
- C. It stores the project’s Python packages
- D. It activates the virtual environment

**Answer:** B


### Short-Answer Questions

**13. Explain the difference between Generative AI and an LLM.**

**Answer:**

Generative AI is a broad category of artificial intelligence that creates new content, including text, images, audio, video, and code. An LLM is a language-focused type of Generative AI that primarily understands and generates language tokens.

Some multimodal systems connect an LLM with specialized components to process or generate images, audio, or video. Therefore, all generative LLMs are part of GenAI, but not all GenAI models are LLMs.


**14. Give one text-based GenAI use case and one multimodal GenAI use case.**

**Answer:**

**Text-based use case:**

Provide a headline, predict its clickbait label, explain the prediction, and generate a neutral rewrite.

**Multimodal use case:**

Provide an image with written instructions and ask the model to identify, analyze, and describe its visible content.


**15. Name two major GenAI risks and provide one suitable mitigation for each.**

**Answer:**

**Risk 1: Hallucination**

The model may generate convincing but factually incorrect information.

**Mitigation:** Verify important claims against reliable sources before accepting or using the output.

**Risk 2: Copyright infringement**

Generated content may resemble protected material, or copyrighted data may be used without authorization.

**Mitigation:** Use properly licensed data, track sources, and confirm usage rights before training a model or publishing generated content.



### Result

| Section | Score |
|---|---:|
| Multiple-choice questions | 12/12 |
| Short-answer questions | 2.5/3 |
| **Total** | **14.5/15** |
| **Percentage** | **96.7%** |

----


## Week 1 Status

**Completed ✅**




