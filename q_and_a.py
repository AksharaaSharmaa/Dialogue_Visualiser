def create_chatbot_interface(processed_df):
    """Create a Gemini-powered chatbot interface that can answer questions about the data."""
    import google.generativeai as genai
    import streamlit as st
    import pandas as pd
    
    st.subheader("💬 Chat with Your Data")
    
    # Set up the API key
    GEMINI_API_KEY = "AIzaSyDVh44jf0dulFB1qP8FwnrHb92DY9gBfdU"  # Replace with your actual API key
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Add explanation
    st.markdown("""
    Ask questions about your conversation data and get AI-powered insights.
    
    **Example questions you can ask:**
    - What are the main topics discussed?
    - Who was the most active participant?
    - What was the overall sentiment of the conversation?
    - Summarize the key arguments made in this discussion.
    - Were there any contentious topics?
    """)
    
    # Initialize session state for conversation
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Prepare comprehensive data context for the model
    data_context = ""
    if not processed_df.empty:
        # Basic statistics
        users = processed_df['user'].unique()
        num_messages = len(processed_df)
        time_range = ""
        
        # Get timestamp range if available
        if 'timestamp' in processed_df.columns:
            try:
                earliest = pd.to_datetime(processed_df['timestamp']).min()
                latest = pd.to_datetime(processed_df['timestamp']).max()
                time_range = f"The conversation spans from {earliest.strftime('%Y-%m-%d %H:%M')} to {latest.strftime('%Y-%m-%d %H:%M')}."
            except:
                pass
        
        # User activity data
        user_activity = processed_df['user'].value_counts()
        most_active_user = user_activity.index[0] if not user_activity.empty else "Unknown"
        most_active_count = user_activity.iloc[0] if not user_activity.empty else 0
        
        # Sentiment information if available
        sentiment_info = ""
        if 'sentiment_compound' in processed_df.columns:
            avg_sentiment = processed_df['sentiment_compound'].mean()
            sentiment_label = "positive" if avg_sentiment > 0.05 else ("negative" if avg_sentiment < -0.05 else "neutral")
            
            # Get most positive and negative messages
            if len(processed_df) > 0:
                most_positive_idx = processed_df['sentiment_compound'].idxmax()
                most_negative_idx = processed_df['sentiment_compound'].idxmin()
                
                most_positive_msg = processed_df.loc[most_positive_idx, 'text'][:100] + "..." if len(processed_df.loc[most_positive_idx, 'text']) > 100 else processed_df.loc[most_positive_idx, 'text']
                most_negative_msg = processed_df.loc[most_negative_idx, 'text'][:100] + "..." if len(processed_df.loc[most_negative_idx, 'text']) > 100 else processed_df.loc[most_negative_idx, 'text']
                
                sentiment_info = f"""The overall sentiment is {sentiment_label} ({avg_sentiment:.2f}).
                Most positive message: "{most_positive_msg}"
                Most negative message: "{most_negative_msg}"
                """
        
        # Topics/Keywords information if available
        topics_info = ""
        if 'keywords' in processed_df.columns:
            all_keywords = []
            for keyword_list in processed_df['keywords']:
                if isinstance(keyword_list, list):
                    all_keywords.extend(keyword_list)
            
            if all_keywords:
                top_keywords = pd.Series(all_keywords).value_counts().head(10).index.tolist()
                topics_info = f"The main topics/keywords appear to be: {', '.join(top_keywords)}."
        
        # Entity information if available
        entities_info = ""
        if 'entities' in processed_df.columns:
            all_entities = []
            for entity_list in processed_df['entities']:
                if isinstance(entity_list, list):
                    all_entities.extend(entity_list)
            
            if all_entities:
                top_entities = pd.Series(all_entities).value_counts().head(5).index.tolist()
                entities_info = f"Key entities mentioned: {', '.join(top_entities)}."
        
        # Reply structure information if available
        conversation_structure = ""
        if 'reply_to' in processed_df.columns:
            replies = processed_df['reply_to'].dropna().count()
            reply_percent = (replies / len(processed_df)) * 100 if len(processed_df) > 0 else 0
            max_replies = processed_df['reply_to'].value_counts().max() if not processed_df['reply_to'].value_counts().empty else 0
            
            conversation_structure = f"""
            This is a conversational dataset with {replies} replies ({reply_percent:.1f}% of messages are replies).
            The most responded-to message received {max_replies} replies.
            """
        
        # Create comprehensive context summary
        data_context = f"""
        This conversation dataset contains {num_messages} messages from {len(users)} users/sources.
        {time_range}
        
        User activity: {most_active_user} was most active with {most_active_count} messages.
        
        {sentiment_info}
        
        {topics_info}
        
        {entities_info}
        
        {conversation_structure}
        """
        
        # Add sample messages for context
        if len(processed_df) > 0:
            # Get first few messages and a sample from the middle
            first_messages = processed_df.head(3)['text'].tolist()
            middle_idx = len(processed_df) // 2
            middle_messages = processed_df.iloc[middle_idx:middle_idx+2]['text'].tolist() if middle_idx + 2 <= len(processed_df) else []
            
            sample_messages = first_messages + middle_messages
            truncated_samples = [f"- {msg[:100]}..." if len(msg) > 100 else f"- {msg}" for msg in sample_messages]
            
            data_context += f"\n\nHere are some sample messages from the conversation:\n" + "\n".join(truncated_samples)
    
    # Input for new question
    prompt = st.chat_input("Ask a question about your data")
    if prompt:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize Gemini model
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # Get previous conversation for context
                    conversation_history = []
                    for msg in st.session_state.chat_history[:-1]:  # Exclude the most recent user message
                        role = "user" if msg["role"] == "user" else "model"
                        conversation_history.append({"role": role, "parts": [msg["content"]]})
                    
                    # Create a new conversation including the data context
                    if conversation_history:
                        chat = model.start_chat(history=conversation_history)
                        full_prompt = f"""Given this data context about a conversation dataset: 
                        
                        {data_context}
                        
                        User question: {prompt}
                        
                        Provide a helpful, accurate response based on this data context. Don't make up information that isn't in the data.
                        If you don't have enough information to answer confidently, say so and explain what additional data would help.
                        """
                    else:
                        chat = model.start_chat()
                        full_prompt = f"""You are an AI assistant analyzing conversation data. Here's the context about the dataset:
                        
                        {data_context}
                        
                        User question: {prompt}
                        
                        Provide a helpful, accurate response based on this data context. Don't make up information that isn't in the data.
                        If you don't have enough information to answer confidently, say so and explain what additional data would help.
                        """
                    
                    # Get response from Gemini
                    response = chat.send_message(full_prompt)
                    response_text = response.text
                    
                    # Display response
                    st.write(response_text)
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    
    # Add option to clear chat history
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()