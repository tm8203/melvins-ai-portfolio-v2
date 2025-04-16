import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from textblob import TextBlob
import os
import streamlit.components.v1 as components
import platform

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Melvin Tejada's AI Portfolio",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.write("🔍 Python version:", platform.python_version())
st.write("📦 pandas version:", pd.__version__)
st.write("📦 nltk version:", __import__('nltk').__version__)


if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"  # Default state

if "allow_sidebar_open" not in st.session_state:
    st.session_state.allow_sidebar_open = True  # Track manual sidebar opening

# Set initial sidebar state based on allow_sidebar_open
#st.set_page_config(
#    page_title="Melvin Tejada's AI Portfolio",
#    layout="wide",
#    initial_sidebar_state="collapsed" if st.session_state.allow_sidebar_open else "expanded"
#)

#st.set_page_config(
#    page_title="Melvin Tejada's AI Portfolio",
#    layout="wide",
#    initial_sidebar_state="collapsed" if st.session_state.allow_sidebar_open else "expanded"
#)

# Initialize sidebar state
#if "sidebar_state" not in st.session_state:
    #st.session_state.sidebar_state = "collapsed"  # Default state

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"  # Start collapsed on first load
    st.session_state.allow_sidebar_open = True  # Track manual sidebar opening

# Inject custom CSS
st.markdown("""
    <style>
        /* Left justify button text */
        .stButton button {
            text-align: left !important;
        }
    </style>
""", unsafe_allow_html=True)


#data = pd.read_csv("Portfolio_Home_Files/synthetic_cloud_pricing_dataset.csv")
try:
    data = pd.read_csv("Portfolio_Home_Files/synthetic_cloud_pricing_dataset.csv")
except Exception as e:
    st.error(f"⚠️ Could not load dataset: {e}")
    data = pd.DataFrame()  # fallback so rest of app still loads


# Sentiment Analysis Function
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Sidebar Structure with session state for navigation
st.sidebar.title("Melvin Tejada's AI Portfolio")
if "page" not in st.session_state:
    st.session_state.page = "about-me"

#def navigate(page_name):
    #if st.session_state.page != page_name:
        #st.session_state.page = page_name
        #st.query_params.update({"page": page_name})  # Update query params to force a refresh

def navigate(page_name):
    if st.session_state.page != page_name:
        st.session_state.page = page_name
        st.session_state.sidebar_state = "collapsed"  # Collapse sidebar on button click
        st.session_state.allow_sidebar_open = True  # Allow manual opening again
        st.session_state.force_refresh = True  # Force refresh to apply changes




#def navigate(page_name):
    #if st.session_state.page != page_name:
        #st.session_state.page = page_name
        #st.session_state.sidebar_state = "collapsed"  # Collapse sidebar after button click
        #st.session_state.needs_rerun = True  # Use flag instead of st.rerun()




st.sidebar.button("About Me", on_click=navigate, args=("about-me",))
st.sidebar.markdown("## Demo and Deployment walk-through of multi-cloud AI/ML/DL models with Terraform and Docker", unsafe_allow_html=True)
st.sidebar.button("DEMO: Deep Learning Fraud Detection Model", on_click=navigate, args=("fraud-model",))
st.sidebar.markdown("## Scenario:<br>Analyze cloud spend by customer segment using AI/ML/DL", unsafe_allow_html=True)
st.sidebar.button("First I Create Synthetic Data. Then...", on_click=navigate, args=("creating-synthetic-data",))
st.sidebar.button("[**interactive**] Use Scalar Regression to Forecast Spend", on_click=navigate, args=("scalar-regression",))
st.sidebar.button("[**interactive**] Use a Variational Autoencoder (VAE) to Detect Anomalies", on_click=navigate, args=("vae-anomaly-detection",))
st.sidebar.button("[**interactive**] Use Natural Language Processing (NLP) to Analyze Customer Insights", on_click=navigate, args=("nlp-customer-insights",))
st.sidebar.markdown("## More Fun AI and Deep Learning Samples")
st.sidebar.button("Create Your Own Picture Filter with Neural Style Transfer (NST)", on_click=navigate, args=("nst-filters",))
st.sidebar.button("Detect Image Features with a Convolutional Neural Network (CNN)", on_click=navigate, args=("cnn-feature-detection",))
st.sidebar.button("[**interactive**] Analyze Sound with Spectrogram Insights", on_click=navigate, args=("sound-analysis",))
st.sidebar.markdown("## Research")
st.sidebar.button("Psychology + Technology Research", on_click=navigate, args=("research",))


# Display relevant section
page = st.session_state.page

if page == "about-me":
    st.title("About Me")
    
    # Display your photo
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Portfolio_Home_Files/photo.jpg", caption="Melvin Tejada", width=150)  # Adjust the width as needed
    
    # Display your blurb in the second column
    with col2:
        st.markdown("""
            Welcome to my AI portfolio! I'm **Melvin Tejada**. I recently spent a year sharpening my skills at the intersection of psychology - specifically technical teams and their motivation and performance; technology - specifically artificial intelligence and deep learning; and product management - as a effective mechanism for tying it all together to deliver real tech. solutions! I'm now ready for my next gig where I hope to lead in a technical product management, 
            program management, or analytics role. I focus on delivering innovative solutions that bridge cutting-edge AI with impactful business outcomes.  
            
            Explore this portfolio (starting with the ">"/menu symbol in the upper left corner) to see a small sample of my work with data and AI/ML/DL.
            
            **_Pro Tip: this portfolio pairs well with a sense of curiosity_** :)
            
            Portfolio sections:  
            - 1. A scenario analyzing cloud spend with a series of **interactive** AI models (~ one minute or less each)  
            - 2. More **interactive**  models that create image filters, detect images, and analyze sound visuaully (also each a minute or less)!
            - 3. Research Papers and Demos
            
            Have fun!
            
            Feel free to connect via LinkedIn (link below) or email me at **tejada.melvin@gmail.com**. 
            """, unsafe_allow_html=True)
    
        st.write("[LinkedIn](https://www.linkedin.com/in/melvin-tejada/) | [Technical Resume](https://github.com/tm8203/melvin-ai-portfolio/blob/main/melvins-resume-2025.pdf)") | [Non-Technical Resume](https://github.com/tm8203/melvin-ai-portfolio/blob/main/melvins-resume-2025.pdf)")

elif page == "fraud-model":
    st.title("AI Fraud Model Demo")
    st.write("**Overview:** This section showcases a video of my deep learning fraud detection model; the second half shows how to deploy such a model using **Terraform, Docker, and AWS**.")

    # Embed the demo video
    st.subheader("🎥 Demo Video")
    video_url = "https://www.youtube.com/embed/ESKxrOGs6cw"  # Change this to match your video ID
    st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <iframe width="900" height="600" src="{video_url}" frameborder="0" allowfullscreen></iframe>
    </div>
    """,
    unsafe_allow_html=True
)
      
           
    # Supporting Documents and Architecture
    st.subheader("📄 Supporting Documents and Architecture")
    st.write("[View Full Code on Github](https://github.com/tm8203/melvin-ai-portfolio/blob/main/model/model.py)")
    st.write("[Deployment Documentation](https://github.com/tm8203/melvin-ai-portfolio/blob/main/deployment/Deployment%20Documentation%20MTejada.%202025.pdf)")
    st.write("[Coming Soon: Terraform Scripts](#)")

    st.subheader("🚀 Deployment Walkthrough")
    st.write("""
    This section showcases how AI models like this one can be deployed in the cloud using Terraform, AWS, and Docker.  
    Rather than manually spinning up servers, this approach defines infrastructure as code (IaC), making it scalable and reproducible.

    🔹 **What’s Next?** Expanding with an **interactive API** for submitting test data.
    """)


    st.image("https://raw.githubusercontent.com/tm8203/melvin-ai-portfolio/main/model/architecture.jpg", 
         caption="AI Model Deployment Architecture", 
         use_container_width=True)





elif page == "creating-synthetic-data":
    st.title("Creating Synthetic Data")
    st.write("**Description:** To simulate a real business scenario of analyzing pricing and spend on cloud services, I fully synthesized this dataset of 500 customer accounts using custom Python scripts. The dataset generates realistic cloud data simulating usage patterns, service usage, and satisfaction metrics (data sample below). The following three modules further develop this pricing scenario, to include spend forecasts, detecting anomalies in the data, and analyzing customer feedback with a Natural Language Processing (NLP) model.")
    st.dataframe(data.head(10))
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/blob/main/Portfolio_Home_Files/generate_dataset.py)") # Add a link to GitHub for the code


elif page == "scalar-regression":
    st.title("Scalar Regression for Spend Forecast")
    st.write("**Opportunity:** Predict future cloud resource usage trends to inform pricing agreements.")
    st.write("**AI Solution:** I developed and trained a scalar regression model to use historical time-series data for analysis.")
    segment = st.selectbox("Select Customer Segment:", data["Customer_Segment"].unique())
    filtered_data = data[data["Customer_Segment"] == segment]
    st.write(f"### Spend Forecast for {segment} Customers")
    x = list(range(1, 13))
    y = filtered_data["Monthly_Cloud_Spend"].sample(12, random_state=42).sort_values().values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker='o')
    ax.set_title(f"Spend Forecast ({segment})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Projected Spend ($)")
    st.pyplot(fig)
    st.write("[View Full Code on GitHub - updated notebook coming soon; all others on avail. on github]") # Add a link to GitHub for the code

elif page == "vae-anomaly-detection":
    st.title("Variational Autoencoder (VAE) for Anomaly Detection")
    st.write("**Opportunity:** Identify inefficiencies or unusual trends in resource usage to identify savings or optimizations.")
    st.write("**AI Solution:** I developed and trained a VAE to detect anomalies in usage patterns, aiding cost optimization.")
    threshold_multiplier = st.slider("Set Threshold Multiplier (mean + (multiplier x standard deviation; lower = more sensitive/more flagged; higher = less sensitive/fewer flagged (extreme)):", 1.0, 3.0, 1.5)
    threshold = data["Monthly_Cloud_Spend"].mean() + threshold_multiplier * data["Monthly_Cloud_Spend"].std()
    anomalies = data[data["Monthly_Cloud_Spend"] > threshold]
    st.write(f"Detected {len(anomalies)} anomalies (Spend > ${threshold:,.2f}).")
    st.dataframe(anomalies)
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/blob/main/VAE/VAE_e89_Melvin_Tejada_HW7.ipynb)") # Add a link to GitHub for the code
    
elif page == "nlp-customer-insights":
    st.title("Natural Language Processing (NLP) for Customer Insights")
    st.write("**Opportunity:** Extract themes and sentiments from customer feedback to improve product features and services.")
    st.write("**AI Solution:** I developed and trained an NLP model to offer text classification, sentiment analysis, and clustering.")

    # Always Visible: Feedback Clustering Visualization
    st.write("### Feedback Clustering")
    chart_data = pd.DataFrame({
        'Topic': ['Pricing', 'Usability', 'Support'],
        'Frequency': [120, 90, 60]
    })
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x='Frequency',
        y=alt.Y('Topic', sort='-x'),
        color='Topic'
    ).properties(
        title="Feedback Clustering by Topic"
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # Add the guidance text back
    st.write("Enter your own sample customer feedback, then click the **Analyze Sentiment** button for results:")

    # Inject CSS to hide the "Press Ctrl+Enter to apply" hint
    st.markdown("""
        <style>
            /* Hide Streamlits Ctrl+Enter message */
            .stTextArea div[data-testid="stMarkdownContainer"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # User Input for Sentiment Analysis
    user_feedbacks = st.text_area("", height=150)

    # Process sentiment analysis only when button is clicked
    if st.button("Analyze Sentiment"):
        if user_feedbacks.strip():  # Ensure input is not empty
            feedback_list = user_feedbacks.split("\n")
            results = [{"Feedback": feedback, "Sentiment": analyze_sentiment(feedback)} for feedback in feedback_list if feedback.strip()]
            results_df = pd.DataFrame(results)
            
            # Show Sentiment Analysis Results Below Button
            st.write("### Sentiment Analysis Results")
            st.dataframe(results_df)
        else:
            st.warning("Please enter some text before clicking 'Analyze Sentiment'.")
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/blob/main/NLP/NLP_e89-Melvin-Tejada-HW8.ipynb)") # Add a link to GitHub for the code




    #this version uses the default ctrl+enter to interact with the module creating a poor mobile user experience
    #if user_feedbacks:
        #feedback_list = user_feedbacks.split("\n")
        #results = [{"Feedback": feedback, "Sentiment": analyze_sentiment(feedback)} for feedback in feedback_list if feedback.strip()]
        #results_df = pd.DataFrame(results)
        #st.write("### Sentiment Analysis Results")
        #st.dataframe(results_df)


elif page == "nst-filters":
    st.title("Create Filters with Neural Style Transfer (NST)")
    st.write("**Overview:** BYOF - Bring Your Own Filter! This project demonstrates Neural Style Transfer (NST) by combining the content of a personal photo with the artistic style of a Kandinsky painting. I applied deep learning to create unique, visually engaging filters for images.")
    st.write("**AI Solution:** My NST model uses deep learning to blend the structure of one image with the artistic style of another.")

    # Display the original photo
    st.subheader("Original Photo")
    st.image("Portfolio_Home_Files/photo.jpg", caption="Original Photo", width=400)

    # Display the Kandinsky painting
    st.subheader("Artistic Style Reference")
    st.image("NST/vii.jpg", caption="Kandinsky Painting", width=400)

    # Display the result
    st.subheader("Resulting Image with Neural Style Transfer")
    st.image("NST/result.png", caption="Styled Image", width=400)

    # Add a link to GitHub for the code
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/tree/main/NST)")


elif page == "cnn-feature-detection":
    st.title("Image Feature Detection with CNNs")
    st.write("**Overview:** This project demonstrates how Convolutional Neural Networks (CNNs) are used to detect features in images, such as edges, patterns, and textures; this can help with image classification or other vision tasks.")
    st.write("**AI Solution:** I applied my CNN model to a lion image to highlight its ability to detect features e.g., edges and textures, so that these fundamental patterns can be used to build higher-level understanding for tasks like image classification, object detection, and automated decision-making in real-world AI applications.")

    # Display the original lion image
    st.subheader("Original Image")
    st.image("CNN/lion.jpg", caption="Original Lion Image", width=400)

    # Display the feature maps (Conv2D output)
    st.subheader("Feature Maps from Convolutional Layers (represent extracted spatial features)")
    st.image("CNN/conv2d_18.png", caption="Conv2D Layer 18", use_container_width=True)
    st.image("CNN/conv2d_19.png", caption="Conv2D Layer 19", use_container_width=True)

    # Display the max pooling output
    st.subheader("Feature Maps after Max Pooling Layers (summarize the most prominent features) ")
    st.image("CNN/max_pooling2d_18.png", caption="Max Pooling Layer 18", use_container_width=True)
    st.image("CNN/max_pooling2d_19.png", caption="Max Pooling Layer 19", use_container_width=True)

    # Add a link to GitHub for the code
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/tree/main/CNN)")


elif page == "sound-analysis":
    st.title("Analyze Sound with Spectrogram Insights")
    st.write("**Overview:** This project demonstrates how audio data can be transformed into visual spectrograms to reveal patterns in frequency and amplitude over time. Spectrograms allow us to analyze sound in a structured, visual format, enabling insights into the characteristics of different audio signals.")
    st.write("**AI Solution:** My model converts audio files into spectrograms to highlight features such as pitch, energy levels, and temporal patterns. These visualizations provide the foundation for tasks like sound classification, speech recognition, and audio anomaly detection, enabling AI to understand and process audio data in real-world applications.")

    # Select a category
    st.subheader("Explore Audio Files by Category")
    category = st.selectbox("Choose a sound category:", ["dog", "eight", "happy"])

    # Audio folder path
    audio_folder = f"audio/{category}/"
    try:
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    except FileNotFoundError:
        st.error(f"Audio folder not found: {audio_folder}")
        audio_files = []

    # Select an audio file
    if audio_files:
        selected_file = st.selectbox("Choose an audio file:", audio_files)
        file_path = os.path.join(audio_folder, selected_file)

        # Play the selected audio file
        st.audio(file_path, format="audio/wav")

        # Generate and display the spectrogram
        st.subheader("Spectrogram of the Selected Audio")
        import matplotlib.pyplot as plt
        from scipy.io import wavfile
        import numpy as np

        try:
            sample_rate, audio_data = wavfile.read(file_path)
            plt.figure(figsize=(10, 4))
            plt.specgram(audio_data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis")
            plt.title(f"Spectrogram of {selected_file}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label="Intensity (dB)")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error generating spectrogram: {e}")
    else:
        st.warning("No audio files found in the selected category.")

    # Add a link to GitHub for the code
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/tree/main/Sound)")

#elif page == "research-demos":
#    st.title("Below is a sample of my research combining Psychology and Technology (all recieved a letter grade of A).")
#    st.write("[PAPER: Performance Management, Organizational Effectiveness, and Disruptive Technology](https://github.com/tm8203/melvin-ai-portfolio/blob/main/Performance%20Mgmt.%20Org.%20Eff.%20and%20Disruptive%20Technology.%20MTejada%2012.04.23.pdf)")
#    st.write("[PAPER: Technology and Trust in the Workplace: A literature Review on the Benefits and Challenges on Workplace Relationships](https://github.com/tm8203/melvin-ai-portfolio/blob/main/Technology%20and%20Trust%20in%20the%20Workplace.%20MTejada.%207.8.24.pdf)")
#    st.write("[PAPER: Trust and Technology Adoption: How Relationships Influence IT Adoption in the Public Sector](https://github.com/tm8203/melvin-ai-portfolio/blob/main/Trust%20and%20Technology%20Adoption.%20MTejada.%208.6.24.pdf)")
#    st.write("[PAPER: Harnessing Tech Talent: The Science Behind Selecting Top Software Sales Representatives](https://github.com/tm8203/melvin-ai-portfolio/blob/main/Harnessing%20Tech%20Talent.%20MTejada.%204.13.24.pdf)")
#    st.write("[DEMO: Employee Fraud Detection: A Deep Learning Model Harnessing Psychology Research to Detect Fraud - Coming Soon]()")

#elif page == "research-demos":
#    st.title("Here is a sample of some of my research combining Psychology and Technology (all received As).")
    
    # Use st.markdown to enable HTML rendering and underline only the titles
#    st.markdown('<ul><li><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Performance%20Mgmt.%20Org.%20Eff.%20and%20Disruptive%20Technology.%20MTejada%2012.04.23.pdf"><u>PAPER: Performance Management, Organizational Effectiveness, and Disruptive Technology</u></a></li></ul>', unsafe_allow_html=True)
#    st.markdown('<ul><li><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Technology%20and%20Trust%20in%20the%20Workplace.%20MTejada.%207.8.24.pdf"><u>PAPER: Technology and Trust in the Workplace: A literature Review on the Benefits and Challenges on Workplace Relationships</u></a></li></ul>', unsafe_allow_html=True)
#    st.markdown('<ul><li><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Trust%20and%20Technology%20Adoption.%20MTejada.%208.6.24.pdf"><u>PAPER: Trust and Technology Adoption: How Relationships Influence IT Adoption in the Public Sector</u></a></li></ul>', unsafe_allow_html=True)
#    st.markdown('<ul><li><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Harnessing%20Tech%20Talent.%20MTejada.%204.13.24.pdf"><u>PAPER: Harnessing Tech Talent: The Science Behind Selecting Top Software Sales Representatives</u></a></li></ul>', unsafe_allow_html=True)
#    st.markdown('<ul><li><a href="#"><u>DEMO: Employee Fraud Detection: A Deep Learning Model Harnessing Psychology Research to Detect Fraud - Coming Soon</u></a></li></ul>', unsafe_allow_html=True)

elif page == "research":
    st.title("Below is a sample of my research combining Psychology and Technology (all received a letter grade of A).")

    # Underline only the titles, not the "PAPER" or "DEMO"
    st.markdown('<p>PAPER: <u><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Performance%20Mgmt.%20Org.%20Eff.%20and%20Disruptive%20Technology.%20MTejada%2012.04.23.pdf">Performance Management, Organizational Effectiveness, and Disruptive Technology</a></u></p>', unsafe_allow_html=True)
    st.markdown('<p>PAPER: <u><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Technology%20and%20Trust%20in%20the%20Workplace.%20MTejada.%207.8.24.pdf">Technology and Trust in the Workplace: A literature Review on the Benefits and Challenges on Workplace Relationships</a></u></p>', unsafe_allow_html=True)
    st.markdown('<p>PAPER: <u><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Trust%20and%20Technology%20Adoption.%20MTejada.%208.6.24.pdf">Trust and Technology Adoption: How Relationships Influence IT Adoption in the Public Sector</a></u></p>', unsafe_allow_html=True)
    st.markdown('<p>PAPER: <u><a href="https://github.com/tm8203/melvin-ai-portfolio/blob/main/Harnessing%20Tech%20Talent.%20MTejada.%204.13.24.pdf">Harnessing Tech Talent: The Science Behind Selecting Top Software Sales Representatives</a></u></p>', unsafe_allow_html=True)


# Ensure sidebar collapse logic applies correctly
#if st.session_state.get("force_refresh", False):
#    st.session_state.force_refresh = False  # Reset flag
#    st.rerun()  # Safe place to trigger refresh




