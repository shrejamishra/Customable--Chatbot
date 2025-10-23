# app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os
import random
import string
import re

# Try to import and download NLTK resources but provide fallbacks if they fail
try:
    import nltk
    # Direct download approach - more reliable
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except Exception as e:
    st.warning(f"NLTK resources could not be loaded. Using simplified text processing instead. Error: {e}")
    nltk_available = False

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Chat",
    page_icon="💬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f7ff;
        border-left: 5px solid #2196F3;
    }
    .bot-message {
        background-color: #f0f0f0;
        border-left: 5px solid #4CAF50;
    }
    .sentiment-badge {
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        width: fit-content;
    }
    .positive {
        background-color: rgba(76, 175, 80, 0.2);
        color: #2e7d32;
    }
    .negative {
        background-color: rgba(244, 67, 54, 0.2);
        color: #c62828;
    }
    .neutral {
        background-color: rgba(158, 158, 158, 0.2);
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pos_scores' not in st.session_state:
    st.session_state.pos_scores = []
if 'neg_scores' not in st.session_state:
    st.session_state.neg_scores = []
if 'neu_scores' not in st.session_state:
    st.session_state.neu_scores = []
if 'compound_scores' not in st.session_state:
    st.session_state.compound_scores = []
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False
if 'words_count' not in st.session_state:
    st.session_state.words_count = {}
if 'bot_mode' not in st.session_state:
    st.session_state.bot_mode = "Standard"

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment class
def get_sentiment_class(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment

# Function to preprocess text using NLTK if available, otherwise use regex
def preprocess_text(text):
    if nltk_available:
        try:
            # Convert to lowercase and tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove punctuation and stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
            
            # Lemmatize words
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            return tokens
        except Exception as e:
            # Fallback if any NLTK process fails
            st.warning(f"NLTK processing failed. Using simple tokenization instead. Error: {e}")
            return simple_tokenize(text)
    else:
        return simple_tokenize(text)

# Simple tokenization fallback using regex
def simple_tokenize(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split by whitespace
    tokens = text.split()
    
    # Remove common stopwords (simplified list)
    common_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                       'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                       'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                       'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
                       'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
                       'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
                       'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                       'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                       'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                       'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                       'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                       'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
                       'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                       'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
    
    filtered_tokens = [word for word in tokens if word not in common_stopwords and len(word) > 1]
    
    return filtered_tokens

# Function to update word count
def update_word_count(text):
    tokens = preprocess_text(text)
    for token in tokens:
        if token in st.session_state.words_count:
            st.session_state.words_count[token] += 1
        else:
            st.session_state.words_count[token] = 1

# Simple rule-based chatbot function
def generate_response(user_input, sentiment):
    # Preprocess the input
    tokens = preprocess_text(user_input)
    
    # Dictionary of responses based on sentiment
    responses = {
        "positive": [
            "I'm glad to hear that! Tell me more.",
            "That sounds great! What else is on your mind?",
            "Wonderful! Is there anything specific you'd like to talk about?",
            "That's positive news! How does that make you feel?",
            "Excellent! Would you like to elaborate on that?"
        ],
        "negative": [
            "I'm sorry to hear that. Would you like to talk about it more?",
            "That sounds challenging. How are you coping with it?",
            "I understand that must be difficult. Is there anything that might help?",
            "I appreciate you sharing that. What do you think could improve the situation?",
            "That's tough. How long has this been going on?"
        ],
        "neutral": [
            "Interesting. Could you tell me more about that?",
            "I see. What are your thoughts on this matter?",
            "Thank you for sharing. Is there anything specific you'd like to discuss?",
            "I understand. Where would you like to take this conversation?",
            "Let's explore that further. What aspects interest you most?"
        ]
    }
    
    # Determine the dominant sentiment
    if sentiment["compound"] > 0.05:
        sentiment_type = "positive"
    elif sentiment["compound"] < -0.05:
        sentiment_type = "negative"
    else:
        sentiment_type = "neutral"
    
    # Educational mode provides information about sentiment analysis
    if st.session_state.bot_mode == "Educational":
        return f"I analyzed your message and detected a {sentiment_type} sentiment (compound score: {sentiment['compound']:.2f}). Positive: {sentiment['pos']:.2f}, Negative: {sentiment['neg']:.2f}, Neutral: {sentiment['neu']:.2f}. This analysis is based on the VADER sentiment analysis model, which uses a lexicon of words rated for valence."
    
    # Therapeutic mode focuses on emotions
    elif st.session_state.bot_mode == "Therapeutic":
        if sentiment_type == "positive":
            return "I notice you're expressing positive emotions. It's great that you're feeling this way. Would you like to reflect on what's contributing to these good feelings?"
        elif sentiment_type == "negative":
            return "I sense some negative emotions in your message. It's okay to feel this way. Would you like to talk more about what might be causing these feelings?"
        else:
            return "Your message seems emotionally neutral. How are you actually feeling right now? Sometimes it helps to name our emotions."
    
    # Mirror mode reflects the user's sentiment
    elif st.session_state.bot_mode == "Mirror":
        if sentiment_type == "positive":
            return "Your message has a positive tone, and that makes me feel positive too! Let's continue this uplifting conversation."
        elif sentiment_type == "negative":
            return "I'm picking up on some negative feelings in your message. I want you to know that it's okay to express these emotions."
        else:
            return "Your message seems fairly neutral in tone. I'm here to chat about whatever you'd like, whether it's casual conversation or something more significant."
    
    # Default mode (Standard)
    else:
        # Choose a random response based on sentiment
        return random.choice(responses[sentiment_type])

# Function to start chat
def start_chat():
    st.session_state.chat_active = True
    
# Function to end chat
def end_chat():
    st.session_state.chat_active = False
    
# Function to reset chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.pos_scores = []
    st.session_state.neg_scores = []
    st.session_state.neu_scores = []
    st.session_state.compound_scores = []
    st.session_state.words_count = {}

# Function to export chat
def export_chat():
    df = pd.DataFrame({
        'Message': st.session_state.history,
        'Positive': st.session_state.pos_scores,
        'Negative': st.session_state.neg_scores,
        'Neutral': st.session_state.neu_scores,
        'Compound': st.session_state.compound_scores
    })
    return df

# Title
st.title("💬 Sentiment Analysis Chat")
st.markdown("A self-contained chat application with real-time sentiment analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Bot mode selection
    st.session_state.bot_mode = st.selectbox(
        "Bot Personality Mode", 
        ["Standard", "Educational", "Therapeutic", "Mirror"],
        index=["Standard", "Educational", "Therapeutic", "Mirror"].index(st.session_state.bot_mode)
    )
    
    # Mode descriptions
    if st.session_state.bot_mode == "Standard":
        st.info("Standard mode provides natural conversational responses based on your message sentiment.")
    elif st.session_state.bot_mode == "Educational":
        st.info("Educational mode explains the sentiment analysis of your messages.")
    elif st.session_state.bot_mode == "Therapeutic":
        st.info("Therapeutic mode focuses on emotional aspects of your messages.")
    else:  # Mirror
        st.info("Mirror mode reflects your message sentiment in its responses.")
    
    # Visualization options
    st.header("Visualization")
    chart_type = st.radio("Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Word Cloud"])
    
    # Chat actions
    st.header("Chat Actions")
    if not st.session_state.chat_active:
        st.button("Start Chat", on_click=start_chat, type="primary")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.button("End Chat", on_click=end_chat)
        with col2:
            st.button("Reset Chat", on_click=reset_chat)
    
    # Export options
    if st.session_state.history:
        st.header("Export Data")
        export_format = st.radio("Export Format", ["CSV", "Excel"])
        
        if st.button("Export Chat Data"):
            df = export_chat()
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="chat_sentiment_data.csv",
                    mime="text/csv"
                )
            else:
                try:
                    # Create Excel file
                    df.to_excel("chat_sentiment_data.xlsx", index=False)
                    with open("chat_sentiment_data.xlsx", "rb") as f:
                        excel_data = f.read()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="chat_sentiment_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    # Clean up the file
                    if os.path.exists("chat_sentiment_data.xlsx"):
                        os.remove("chat_sentiment_data.xlsx")
                except Exception as e:
                    st.error(f"Error creating Excel file: {str(e)}")
                    st.info("Try downloading as CSV instead.")

# Main content area
if not st.session_state.chat_active:
    st.info("Click 'Start Chat' in the sidebar to begin a conversation.")
    
    # Display sample conversation
    with st.expander("See a sample conversation"):
        st.markdown("""
        **User**: I'm really excited about this project!
        
        **Bot**: That sounds great! What else is on your mind?
        
        **User**: I'm struggling with some parts of the implementation though.
        
        **Bot**: I'm sorry to hear that. Would you like to talk about it more?
        
        **User**: I think I just need to understand the sentiment analysis part better.
        
        **Bot**: I understand. Where would you like to take this conversation?
        """)
else:
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if i % 2 == 0:  # User message
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div><strong>You:</strong> {message}</div>
                        <div class="sentiment-badge {get_sentiment_class(st.session_state.compound_scores[i//2])}">
                            Sentiment: {get_sentiment_class(st.session_state.compound_scores[i//2])} ({st.session_state.compound_scores[i//2]:.2f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:  # Bot message
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div><strong>Bot:</strong> {message}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # User input
    with st.container():
        user_input = st.text_area("Message:", key="user_input", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            send_btn = st.button("Send", type="primary")
        
        if send_btn and user_input:
            # Add user message to history
            st.session_state.messages.append(user_input)
            st.session_state.history.append(user_input)
            
            # Analyze sentiment
            sentiment = analyze_sentiment(user_input)
            st.session_state.pos_scores.append(sentiment['pos'])
            st.session_state.neg_scores.append(sentiment['neg'])
            st.session_state.neu_scores.append(sentiment['neu'])
            st.session_state.compound_scores.append(sentiment['compound'])
            
            # Update word count for word cloud
            update_word_count(user_input)
            
            # Get bot response with a spinner
            with st.spinner("Thinking..."):
                time.sleep(0.5)  # Simulate thinking time
                bot_response = generate_response(user_input, sentiment)
            
            # Add bot message to chat
            st.session_state.messages.append(bot_response)
            
            # Clear input
            st.rerun()
    
    # Display sentiment analysis charts
    if st.session_state.history:
        st.header("Sentiment Analysis")
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Message Number': list(range(1, len(st.session_state.history) + 1)),
            'Positive': st.session_state.pos_scores,
            'Negative': st.session_state.neg_scores,
            'Neutral': st.session_state.neu_scores,
            'Compound': st.session_state.compound_scores,
            'Message': st.session_state.history
        })
        
        # Word cloud visualization
        if chart_type == "Word Cloud" and st.session_state.words_count:
            try:
                from wordcloud import WordCloud
                
                st.subheader("Word Cloud of Your Messages")
                
                # Generate word cloud
                wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(st.session_state.words_count)
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                # Show top words
                st.subheader("Top 10 Words")
                top_words = dict(sorted(st.session_state.words_count.items(), key=lambda item: item[1], reverse=True)[:10])
                
                # Create bar chart for top words
                fig = px.bar(
                    x=list(top_words.keys()),
                    y=list(top_words.values()),
                    labels={'x': 'Word', 'y': 'Frequency'},
                    title='Top 10 Words Used'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.error("WordCloud package is not installed. Please run 'pip install wordcloud' to use this feature.")
                st.info("Showing line chart instead...")
                chart_type = "Line Chart"
        
        # Line chart visualization
        if chart_type == "Line Chart":
            fig = px.line(
                df, 
                x='Message Number', 
                y=['Positive', 'Negative', 'Compound'],
                markers=True,
                title='Sentiment Trends',
                hover_data=['Message']
            )
            
        # Bar chart visualization
        elif chart_type == "Bar Chart":
            df_melted = pd.melt(
                df, 
                id_vars=['Message Number', 'Message'], 
                value_vars=['Positive', 'Negative', 'Compound'],
                var_name='Sentiment Type', 
                value_name='Score'
            )
            fig = px.bar(
                df_melted, 
                x='Message Number', 
                y='Score', 
                color='Sentiment Type',
                barmode='group',
                title='Sentiment Analysis by Message',
                hover_data=['Message']
            )
            
        # Area chart visualization
        elif chart_type == "Area Chart":
            fig = px.area(
                df, 
                x='Message Number', 
                y=['Positive', 'Negative'],
                title='Sentiment Area Chart',
                hover_data=['Message']
            )
        
        # Display chart (if not word cloud)
        if chart_type != "Word Cloud":
            # Update layout
            fig.update_layout(
                xaxis_title='Message Number',
                yaxis_title='Sentiment Score',
                legend_title='Sentiment Type',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_pos = sum(st.session_state.pos_scores) / len(st.session_state.pos_scores)
            st.metric("Average Positive", f"{avg_pos:.2f}")
        
        with col2:
            avg_neg = sum(st.session_state.neg_scores) / len(st.session_state.neg_scores)
            st.metric("Average Negative", f"{avg_neg:.2f}")
        
        with col3:
            avg_compound = sum(st.session_state.compound_scores) / len(st.session_state.compound_scores)
            st.metric("Average Compound", f"{avg_compound:.2f}")
        
        with col4:
            dominant_sentiment = "Positive" if avg_pos > avg_neg else "Negative"
            if abs(avg_compound) < 0.05:
                dominant_sentiment = "Neutral"
            st.metric("Overall Sentiment", dominant_sentiment)

# Footer
st.markdown("---")
st.markdown("Sentiment Analysis Chat | Built with Streamlit and VADER Sentiment Analysis")