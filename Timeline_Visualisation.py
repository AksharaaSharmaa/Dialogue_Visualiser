def create_timeline_visualization(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import re
    import streamlit as st
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'text', 'user']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure we have sentiment data, or create placeholder
    if 'sentiment_compound' not in df.columns:
        df['sentiment_compound'] = 0
    
    # Create a figure with subplots - increased spacing between subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.22,  # Increased spacing between subplots
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            "<b>Conversation Timeline</b>", 
            "<b>Sentiment Flow</b>", 
            "<b>Message Frequency</b>"
        )
    )
    
    # Calculate time differences to identify gaps in conversation
    df['time_diff'] = df['timestamp'].diff().fillna(pd.Timedelta(seconds=0))
    
    # Identify significant time gaps (e.g., more than 1 hour)
    time_gap_threshold = pd.Timedelta(hours=1)
    significant_gaps = df[df['time_diff'] > time_gap_threshold].index.tolist()
    
    # Identify conversation segments based on time gaps
    segment_indices = [0] + significant_gaps + [len(df)]
    segments = []
    for i in range(len(segment_indices) - 1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i+1]
        segments.append((start_idx, end_idx))
    
    # Store conversation segments for text output
    segment_data = []
    for i, (start_idx, end_idx) in enumerate(segments):
        if start_idx == end_idx - 1:
            continue  # Skip segments with only one message
        
        start_time = df.loc[start_idx, 'timestamp']
        end_time = df.loc[end_idx - 1, 'timestamp']
        duration = end_time - start_time
        message_count = end_idx - start_idx
        
        segment_users = df.iloc[start_idx:end_idx]['user'].value_counts()
        primary_user = segment_users.index[0] if len(segment_users) > 0 else "Unknown"
        
        segment_data.append({
            "segment_id": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "message_count": message_count,
            "primary_user": primary_user,
            "user_distribution": segment_users.to_dict()
        })
    
    # Extract topics from each message
    topic_keywords = []
    if 'keywords' in df.columns:
        topic_keywords = df['keywords'].tolist()
    elif 'entities' in df.columns:
        topic_keywords = df['entities'].tolist()
    else:
        # Simple keyword extraction if NLP processing wasn't done
        def extract_simple_keywords(text, n=3):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_count = {}
            for word in words:
                if word not in ['the', 'and', 'for', 'that', 'this', 'with', 'you', 'have', 'are']:
                    word_count[word] = word_count.get(word, 0) + 1
            return sorted(word_count, key=word_count.get, reverse=True)[:n]
        
        topic_keywords = df['text'].apply(extract_simple_keywords).tolist()
    
    # Identify topic shifts with improved overlap detection
    topic_shifts = []
    current_topics = set()
    last_shift_idx = -20  # Initialize with a buffer to prevent early shifts
    
    # Store topic shifts for text output
    topic_shift_data = []
    
    for i, keywords in enumerate(topic_keywords):
        # Skip if too close to previous shift to avoid overlap
        if i - last_shift_idx < 15:
            continue
            
        if i == 0:
            current_topics = set(keywords[:3]) if keywords else set()
            continue
            
        new_topics = set(keywords[:3]) if keywords else set()
        # Check if topics have significantly changed
        if len(current_topics) > 0 and len(new_topics) > 0:
            overlap = len(current_topics.intersection(new_topics))
            total = len(current_topics.union(new_topics))
            if overlap / total < 0.3:  # Less than 30% overlap indicates topic shift
                topic_shifts.append(i)
                
                # Store topic shift data
                topic_shift_data.append({
                    "shift_id": len(topic_shift_data) + 1,
                    "timestamp": df.loc[i, 'timestamp'],
                    "message_index": i,
                    "user": df.loc[i, 'user'],
                    "previous_topics": list(current_topics),
                    "new_topics": list(new_topics),
                    "message_text": df.loc[i, 'text'][:100] + ("..." if len(df.loc[i, 'text']) > 100 else "")
                })
                
                current_topics = new_topics
                last_shift_idx = i
    
    # Identify question messages (messages ending with '?')
    questions = df[df['text'].str.strip().str.endswith('?')].index.tolist()
    
    # Identify agreement and disagreement messages with refined patterns
    agreement_patterns = [
        r'\bagree\b', r'\byes\b', r'\bexactly\b', r'\btrue\b', r'\bsure\b', 
        r'\bgood point\b', r'\byou\'re right\b', r'\bi think so\b', r'\bthank you\b'
    ]
    disagreement_patterns = [
        r'\bdisagree\b', r'\bno\b', r'\bnot true\b', r'\bi don\'t think\b', 
        r'\bwrong\b', r'\bnot correct\b', r'\bnot necessarily\b', r'\bbut\b'
    ]
    
    agreement_regex = '|'.join(agreement_patterns)
    disagreement_regex = '|'.join(disagreement_patterns)
    
    agreements = df[df['text'].str.lower().str.contains(agreement_regex, regex=True)].index.tolist()
    disagreements = df[df['text'].str.lower().str.contains(disagreement_regex, regex=True)].index.tolist()
    
    # Identify emotional peaks (messages with very positive or negative sentiment)
    if 'sentiment_compound' in df.columns:
        high_positive = df[df['sentiment_compound'] > 0.6].index.tolist()
        high_negative = df[df['sentiment_compound'] < -0.6].index.tolist()
        emotional_peaks = high_positive + high_negative
    else:
        emotional_peaks = []
    
    # Store interaction patterns for text output
    interaction_data = {
        "questions": len(questions),
        "agreements": len(agreements),
        "disagreements": len(disagreements),
        "emotional_peaks": len(emotional_peaks),
        "question_users": df.loc[questions, 'user'].value_counts().to_dict() if questions else {},
        "agreement_users": df.loc[agreements, 'user'].value_counts().to_dict() if agreements else {},
        "disagreement_users": df.loc[disagreements, 'user'].value_counts().to_dict() if disagreements else {},
        "emotional_peak_users": df.loc[emotional_peaks, 'user'].value_counts().to_dict() if emotional_peaks else {}
    }
    
    # Create colormap for users with distinct colors
    unique_users = df['user'].unique()
    # Color palette with better distinction between colors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    user_colors = {user: colors[i % len(colors)] for i, user in enumerate(unique_users)}
    
    # Add traces for each user's messages - with better spacing between users
    y_spacing = 1.2  # Increased spacing between users
    
    for i, user in enumerate(unique_users):
        user_df = df[df['user'] == user]
        
        # Determine marker symbol based on message characteristics
        symbols = []
        for idx in user_df.index:
            if idx in questions:
                symbols.append('diamond')
            elif idx in agreements:
                symbols.append('square')
            elif idx in disagreements:
                symbols.append('x')
            elif idx in emotional_peaks:
                symbols.append('star')
            else:
                symbols.append('circle')
        
        # Determine marker size based on various factors with consistent scaling
        sizes = []
        for idx in user_df.index:
            size = 14  # Base size increased for better visibility
            
            # Increase size for longer messages
            text_length = len(user_df.loc[idx, 'text'])
            size += min(3 * np.log1p(text_length / 20), 10)  # More subtle size differences
            
            # Increase size for emotional messages
            if 'sentiment_compound' in df.columns:
                sentiment = abs(user_df.loc[idx, 'sentiment_compound'])
                size += sentiment * 5
                
            # Increase size for topic shifts
            if idx in topic_shifts:
                size += 8
                
            sizes.append(size)
        
        # Create hover text with rich information
        hover_texts = []
        for _, row in user_df.iterrows():
            hover_info = f"<b>User:</b> {row['user']}<br>"
            hover_info += f"<b>Time:</b> {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            
            # Add sentiment if available
            if 'sentiment_compound' in df.columns:
                sentiment_val = row['sentiment_compound']
                sentiment_label = "positive" if sentiment_val > 0.05 else ("negative" if sentiment_val < -0.05 else "neutral")
                hover_info += f"<b>Sentiment:</b> {sentiment_label} ({sentiment_val:.2f})<br>"
            
            # Message type info
            idx = row.name
            if idx in questions:
                hover_info += "<b>Type:</b> Question<br>"
            elif idx in agreements:
                hover_info += "<b>Type:</b> Agreement<br>"
            elif idx in disagreements:
                hover_info += "<b>Type:</b> Disagreement<br>"
            elif idx in emotional_peaks:
                hover_info += "<b>Type:</b> Emotional Peak<br>"
            else:
                hover_info += "<b>Type:</b> Regular message<br>"
            
            # Add message text (truncated if too long)
            text = row['text']
            if len(text) > 100:
                text = text[:97] + "..."
            hover_info += f"<b>Message:</b><br>{text}"
            
            hover_texts.append(hover_info)
        
        # Add scatter plot trace for this user's messages
        fig.add_trace(
            go.Scatter(
                x=user_df['timestamp'],
                y=[i * y_spacing] * len(user_df),  # Use y_spacing for better vertical separation
                mode='markers',
                marker=dict(
                    color=user_colors[user],
                    size=sizes,
                    symbol=symbols,
                    line=dict(width=1, color='white'),
                    opacity=0.9
                ),
                name=user,
                hovertext=hover_texts,
                hoverinfo="text",
                hovertemplate="%{hovertext}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Add sentiment flow with enhanced information
    sentiment_data = {}
    if 'sentiment_compound' in df.columns:
        # Calculate rolling average for smoother line
        window_size = max(3, len(df) // 20)  # Dynamic window size based on data length
        df['sentiment_smoothed'] = df['sentiment_compound'].rolling(window=window_size, min_periods=1).mean()
        
        # Store sentiment data for metrics
        sentiment_data = {
            "overall_mean": df['sentiment_compound'].mean(),
            "min_sentiment": df['sentiment_compound'].min(),
            "min_sentiment_time": df.loc[df['sentiment_compound'].idxmin(), 'timestamp'],
            "min_sentiment_user": df.loc[df['sentiment_compound'].idxmin(), 'user'],
            "max_sentiment": df['sentiment_compound'].max(),
            "max_sentiment_time": df.loc[df['sentiment_compound'].idxmax(), 'timestamp'],
            "max_sentiment_user": df.loc[df['sentiment_compound'].idxmax(), 'user'],
            "sentiment_stability": df['sentiment_compound'].std(),
            "sentiment_trend": "upward" if df['sentiment_compound'].corr(pd.Series(range(len(df)))) > 0.1 else 
                             "downward" if df['sentiment_compound'].corr(pd.Series(range(len(df)))) < -0.1 else "stable",
            "user_sentiments": df.groupby('user')['sentiment_compound'].agg(['mean', 'min', 'max', 'std']).to_dict('index')
        }
        
        # Add enhanced hover information for sentiment flow
        sentiment_hover_texts = []
        for i, row in df.iterrows():
            hover_text = f"<b>Time:</b> {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            hover_text += f"<b>User:</b> {row['user']}<br>"
            hover_text += f"<b>Sentiment:</b> {row['sentiment_compound']:.2f}<br>"
            hover_text += f"<b>Smoothed:</b> {row['sentiment_smoothed']:.2f}<br>"
            
            # Detect local trends
            if i > 0 and i < len(df) - 1:
                prev_sentiment = df.iloc[i-1]['sentiment_compound']
                next_sentiment = df.iloc[i+1]['sentiment_compound'] if i < len(df) - 1 else row['sentiment_compound']
                
                if row['sentiment_compound'] > prev_sentiment and row['sentiment_compound'] > next_sentiment:
                    hover_text += "<b>Pattern:</b> Local peak<br>"
                elif row['sentiment_compound'] < prev_sentiment and row['sentiment_compound'] < next_sentiment:
                    hover_text += "<b>Pattern:</b> Local valley<br>"
                elif row['sentiment_compound'] > prev_sentiment:
                    hover_text += "<b>Pattern:</b> Rising<br>"
                elif row['sentiment_compound'] < prev_sentiment:
                    hover_text += "<b>Pattern:</b> Falling<br>"
            
            # Add message snippet
            text = row['text']
            if len(text) > 60:
                text = text[:57] + "..."
            hover_text += f"<b>Message:</b> {text}"
            
            sentiment_hover_texts.append(hover_text)
        
        # Add gradient fill for sentiment flow
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sentiment_smoothed'],
                mode='lines',
                line=dict(width=3, color='rgba(65, 105, 225, 0.9)', shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(65, 105, 225, 0.2)',
                name='Sentiment Flow',
                hoverinfo='text',
                hovertext=sentiment_hover_texts,
                hovertemplate="%{hovertext}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Add markers for emotional peaks
        emotion_markers = pd.DataFrame({
            'timestamp': df.loc[emotional_peaks, 'timestamp'],
            'sentiment': df.loc[emotional_peaks, 'sentiment_compound'],
            'user': df.loc[emotional_peaks, 'user'],
            'text': df.loc[emotional_peaks, 'text'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
        })
        
        if not emotion_markers.empty:
            fig.add_trace(
                go.Scatter(
                    x=emotion_markers['timestamp'],
                    y=emotion_markers['sentiment'],
                    mode='markers',
                    marker=dict(
                        color=emotion_markers['sentiment'].apply(
                            lambda x: 'rgba(255, 0, 0, 0.7)' if x < 0 else 'rgba(0, 255, 0, 0.7)'
                        ),
                        size=12,
                        symbol='star',
                        line=dict(width=1, color='white')
                    ),
                    name='Emotional Peaks',
                    hoverinfo='text',
                    hovertext=[
                        f"<b>Emotional Peak</b><br>User: {user}<br>Sentiment: {sentiment:.2f}<br>Message: {text}" 
                        for user, sentiment, text in zip(
                            emotion_markers['user'], 
                            emotion_markers['sentiment'], 
                            emotion_markers['text']
                        )
                    ],
                    hovertemplate="%{hovertext}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Add a zero line for reference
        fig.add_shape(
            type="line",
            x0=df['timestamp'].min(),
            y0=0,
            x1=df['timestamp'].max(),
            y1=0,
            line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Add positive/negative regions
        fig.add_shape(
            type="rect",
            x0=df['timestamp'].min(),
            y0=0,
            x1=df['timestamp'].max(),
            y1=1.2,
            fillcolor="rgba(144, 238, 144, 0.15)",  # Light green for positive
            line=dict(width=0),
            layer="below",
            row=2, col=1
        )
        
        fig.add_shape(
            type="rect",
            x0=df['timestamp'].min(),
            y0=-1.2,
            x1=df['timestamp'].max(),
            y1=0,
            fillcolor="rgba(255, 182, 193, 0.15)",  # Light red for negative
            line=dict(width=0),
            layer="below",
            row=2, col=1
        )
        
        # Add annotations for sentiment trends
        if len(df) >= 5:  # Only add if we have enough data
            start_sentiment = df.iloc[0:min(5, len(df))]['sentiment_compound'].mean()
            end_sentiment = df.iloc[max(0, len(df)-5):len(df)]['sentiment_compound'].mean()
            
            trend_text = None
            if end_sentiment - start_sentiment > 0.3:
                trend_text = "Overall Positive Trend"
            elif start_sentiment - end_sentiment > 0.3:
                trend_text = "Overall Negative Trend"
                
            if trend_text:
                fig.add_annotation(
                    x=df['timestamp'].iloc[len(df)//2],
                    y=0.9,
                    text=trend_text,
                    showarrow=False,
                    font=dict(size=12, color="#333"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#ddd",
                    borderwidth=1,
                    borderpad=4,
                    row=2, col=1
                )
    
    # Add topic shifts with reduced visual clutter
    for idx in topic_shifts:
        shift_time = df.loc[idx, 'timestamp']
        
        # Add a vertical line with subtle styling
        fig.add_shape(
            type="line",
            x0=shift_time,
            y0=-0.5,
            x1=shift_time,
            y1=len(unique_users) * y_spacing - 0.5,
            line=dict(color="rgba(255, 99, 71, 0.5)", width=1.5, dash="dot"),
            row=1, col=1
        )
        
        # Add topic annotation only for selected shifts to reduce clutter
        # Display only every other topic shift annotation
        if topic_keywords[idx] and idx % 2 == 0:  
            new_topic = ", ".join(topic_keywords[idx][:2])  # Limit to 2 keywords
            
            # Calculate a position that avoids overlap
            y_position = (len(unique_users) - 0.5) * y_spacing * 0.8
            
            fig.add_annotation(
                x=shift_time,
                y=y_position,
                text=f"Topic: {new_topic}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="tomato",
                font=dict(size=10, color="white"),
                bgcolor="rgba(255, 99, 71, 0.8)",
                bordercolor="tomato",
                borderwidth=1,
                borderpad=4,
                opacity=0.8,
                ax=0,
                ay=-25,
                row=1, col=1
            )
    
    # Add background shading for conversation segments with subtle styling
    for i, (start_idx, end_idx) in enumerate(segments):
        if start_idx == end_idx - 1:
            continue  # Skip segments with only one message
            
        start_time = df.loc[start_idx, 'timestamp']
        end_time = df.loc[end_idx - 1, 'timestamp']
        
        # Use alternating but very subtle background colors
        color = "rgba(240, 248, 255, 0.2)" if i % 2 == 0 else "rgba(255, 245, 238, 0.2)"
        
        fig.add_shape(
            type="rect",
            x0=start_time,
            y0=-0.5,
            x1=end_time,
            y1=len(unique_users) * y_spacing - 0.5, 
            fillcolor=color,
            line=dict(width=0),
            layer="below",
            row=1, col=1
        )
    
    # Calculate message frequency over time with appropriate binning
    time_span = df['timestamp'].max() - df['timestamp'].min()
    
    # Choose appropriate bin size based on conversation length
    if time_span.total_seconds() < 3600:  # Less than an hour
        bin_size = pd.Timedelta(minutes=5)
        bin_name = "5 minutes"
    elif time_span.total_seconds() < 86400:  # Less than a day
        bin_size = pd.Timedelta(minutes=30)
        bin_name = "30 minutes"
    elif time_span.total_seconds() < 604800:  # Less than a week
        bin_size = pd.Timedelta(hours=4)
        bin_name = "4 hours"
    else:
        bin_size = pd.Timedelta(days=1)
        bin_name = "day"
    
    # Create time bins
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max() + bin_size  # Add one extra bin to include last message
    
    bins = []
    current = start_time
    while current < end_time:
        bins.append(current)
        current += bin_size
    
    # Count messages in each bin
    timestamp_values = df['timestamp'].astype(np.int64) // 10**9  # Convert to Unix timestamp
    bins_values = np.array([b.timestamp() for b in bins])  # Convert datetime to Unix timestamp as numpy array
    hist, bin_edges = np.histogram(timestamp_values, bins=bins_values)
    
    # Calculate bin centers (for positioning the bars)
    bin_centers = [bins[i] + (bins[i+1] - bins[i])/2 for i in range(len(bins)-1)]
    
    # Store frequency data for metrics
    frequency_data = {
        "bin_size": bin_name,
        "max_messages_per_bin": hist.max(),
        "max_messages_time": bins[hist.argmax()] if len(hist) > 0 else None,
        "avg_messages_per_bin": hist.mean() if len(hist) > 0 else 0,
        "empty_bins": sum(1 for h in hist if h == 0),
        "most_active_periods": [
            {
                "start_time": bins[i],
                "end_time": bins[i+1],
                "message_count": hist[i]
            }
            for i in np.argsort(hist)[-3:][::-1] if len(hist) > i
        ]
    }
    
    # Create detailed hover information for frequency bars
    frequency_hover_texts = []
    for i, (bin_start, count) in enumerate(zip(bins[:-1], hist)):
        bin_end = bins[i+1]
        
        # Get messages in this bin
        bin_messages = df[(df['timestamp'] >= bin_start) & (df['timestamp'] < bin_end)]
        
        # Get user breakdown
        user_counts = bin_messages['user'].value_counts()
        user_breakdown = "<br>".join([f"{user}: {count}" for user, count in user_counts.items()])
        
        hover_text = f"<b>Time Period:</b> {bin_start.strftime('%Y-%m-%d %H:%M')} - {bin_end.strftime('%H:%M')}<br>"
        hover_text += f"<b>Total Messages:</b> {count}<br>"
        
        if not bin_messages.empty:
            # Add sentiment info if available
            if 'sentiment_compound' in bin_messages.columns:
                avg_sentiment = bin_messages['sentiment_compound'].mean()
                sentiment_label = "positive" if avg_sentiment > 0.05 else ("negative" if avg_sentiment < -0.05 else "neutral")
                hover_text += f"<b>Average Sentiment:</b> {sentiment_label} ({avg_sentiment:.2f})<br>"
            
            # Add user breakdown
            if not user_counts.empty:
                hover_text += f"<b>Users:</b><br>{user_breakdown}"
        
        frequency_hover_texts.append(hover_text)
    
    # Add message frequency histogram with enhanced information
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist,
            marker=dict(
                color=hist,
                colorscale='Viridis',
                line=dict(width=0.5, color='white'),
                opacity=0.7  # More subtle opacity
            ),
            name='Message Frequency',
            hoverinfo="text",
            hovertext=frequency_hover_texts,
            hovertemplate="%{hovertext}<extra></extra>"
        ),
        row=3, col=1
    )
    
    # Add annotations for peak activity
    if len(hist) > 0:
        peak_idx = hist.argmax()
        fig.add_annotation(
            x=bin_centers[peak_idx],
            y=hist[peak_idx],
            text=f"Peak: {hist[peak_idx]} messages",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#333",
            font=dict(size=10, color="#333"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#ddd",
            borderwidth=1,
            borderpad=4,
            ax=0,
            ay=-25,
            row=3, col=1
        )
    
    # Calculate user activity in bins
    user_activity = {}
    for user in unique_users:
        user_hist, _ = np.histogram(
            df[df['user'] == user]['timestamp'].astype(np.int64) // 10**9,
            bins=bins_values
        )
        user_activity[user] = user_hist
    
    # Add lines showing user activity patterns over time
    if len(bin_centers) > 1:  # Need at least 2 points for lines
        for user, activity in user_activity.items():
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=activity,
                    mode='lines',
                    line=dict(width=1.5, color=user_colors[user], dash='dot'),
                    name=f"{user} Activity",
                    hoverinfo="text",
                    hovertext=[f"{user}: {count} messages" for count in activity],
                    hovertemplate="%{hovertext}<extra></extra>",
                    visible='legendonly'  # Hide by default to reduce clutter
                ),
                row=3, col=1
            )
    
    # Calculate user metrics for text output
    user_counts = df['user'].value_counts()
    text_lengths = df.groupby('user')['text'].apply(lambda x: sum(len(t) for t in x))
    avg_msg_length = text_lengths / user_counts
    
    user_metrics_df = pd.DataFrame({
        'User': user_counts.index,
        'Message_Count': user_counts.values,
        'Percentage': (user_counts / user_counts.sum() * 100).values,
        'Avg_Message_Length': avg_msg_length.values,
        'Total_Characters': text_lengths.values,
        'Questions_Asked': pd.Series({user: sum(1 for idx in df[df['user'] == user].index if idx in questions) for user in unique_users}),
        'Agreements': pd.Series({user: sum(1 for idx in df[df['user'] == user].index if idx in agreements) for user in unique_users}),
        'Disagreements': pd.Series({user: sum(1 for idx in df[df['user'] == user].index if idx in disagreements) for user in unique_users}),
        'Emotional_Peaks': pd.Series({user: sum(1 for idx in df[df['user'] == user].index if idx in emotional_peaks) for user in unique_users})
    })
    
    # Add conversation metrics
    time_span = df['timestamp'].max() - df['timestamp'].min()
    avg_time_between = df['time_diff'].mean()
    
    # Prepare metrics data for return
    metrics = {
        'general': {
            'total_messages': len(df),
            'conversation_duration': str(time_span),
            'avg_time_between_messages': str(avg_time_between),
            'start_time': df['timestamp'].min(),
            'end_time': df['timestamp'].max()
        },
        'users': user_metrics_df.to_dict('records'),
        'segments': segment_data,
        'topic_shifts': topic_shift_data,
        'interactions': interaction_data,
        'sentiment': sentiment_data,
        'frequency': frequency_data
    }
    
    # Update y-axis titles for better clarity
    fig.update_yaxes(
        title_text="Users",
        showticklabels=False,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Sentiment",
        range=[-1.1, 1.1],
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Message Count",
        row=3, col=1
    )
    
    # Add user legend to the timeline (improved positioning)
    for i, user in enumerate(unique_users):
        fig.add_annotation(
            x=df['timestamp'].min(),
            y=i * y_spacing,
            xanchor='right',
            yanchor='middle',
            text=user,
            showarrow=False,
            font=dict(
                family='Arial',
                size=10,
                color=user_colors[user]
            ),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=user_colors[user],
            borderwidth=1,
            borderpad=2,
            row=1, col=1
        )
    
    # Add title annotation with conversation summary - moved above the chart for better visibility
    duration_str = str(time_span).split('.')[0]  # Remove microseconds
    
    fig.add_annotation(
        x=0.5,
        y=1.12,  # Position above the chart
        xref="paper",
        yref="paper",
        text=(
            f"<b>Conversation Analysis</b><br>"
            f"Duration: {duration_str} • "
            f"Messages: {len(df)} • "
            f"Users: {len(unique_users)} • "
            f"Segments: {len(segment_data)} • "
            f"Topic Shifts: {len(topic_shift_data)}"
        ),
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#ddd",
        borderwidth=1,
        borderpad=6,
        width=500
    )
    
    # Update the overall figure layout with improved aesthetics
    fig.update_layout(
        height=800,  # Increased height for better readability
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,  # Position below the chart
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=100, r=50, t=100, b=100),  # Increased margins for annotations
        plot_bgcolor='rgba(250, 250, 250, 0.9)',  # Lighter background
        paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial"
        ),
        showlegend=True
    )
    
    # Update x-axis ranges and formats for consistency across subplots
    time_buffer = time_span * 0.03  # 3% buffer on each side
    x_min = df['timestamp'].min() - pd.Timedelta(seconds=time_buffer.total_seconds())
    x_max = df['timestamp'].max() + pd.Timedelta(seconds=time_buffer.total_seconds())
    
    # Set appropriate date format based on time span
    dtick = None
    if time_span < pd.Timedelta(hours=2):
        tick_format = "%H:%M"
        dtick = "M15"  # 15 minute intervals
    elif time_span < pd.Timedelta(days=1):
        tick_format = "%H:%M"
        dtick = "M30"  # 30 minute intervals
    elif time_span < pd.Timedelta(days=7):
        tick_format = "%m-%d %H:%M"
        dtick = "D1"  # 1 day intervals
    else:
        tick_format = "%Y-%m-%d"
        dtick = "D1"  # 1 day intervals
    
    # Update x-axes with consistent formatting
    fig.update_xaxes(
        range=[x_min, x_max],
        tickformat=tick_format,
        dtick=dtick,
        tickangle=45,
        gridcolor='rgba(220, 220, 220, 0.5)',
        zeroline=False
    )
    
    # Add gridlines to sentiment and frequency plots for readability
    fig.update_yaxes(
        gridcolor='rgba(220, 220, 220, 0.5)',
        zeroline=True,
        zerolinecolor='rgba(128, 128, 128, 0.5)',
        zerolinewidth=1,
        row=2, col=1
    )
    
    fig.update_yaxes(
        gridcolor='rgba(220, 220, 220, 0.5)',
        zeroline=True,
        zerolinecolor='rgba(128, 128, 128, 0.5)',
        zerolinewidth=1,
        row=3, col=1
    )
    
    # Generate text summary of the conversation
    summary_text = f"""
    Conversation Analysis Summary
    
    General Information
    - Total Messages: {len(df)}
    - Duration: {str(time_span).split('.')[0]}
    - Start Time: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}
    - End Time: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}
    - Participants: {len(unique_users)} ({', '.join(unique_users)})
    
    ## User Activity
    | User | Messages | % of Total | Avg Length | Questions | Agreements | Disagreements |
    |------|----------|------------|------------|-----------|------------|---------------|
    """
    
    for _, row in user_metrics_df.sort_values('Message_Count', ascending=False).iterrows():
        summary_text += f"| {row['User']} | {row['Message_Count']} | {row['Percentage']:.1f}% | {row['Avg_Message_Length']:.1f} | {row['Questions_Asked']} | {row['Agreements']} | {row['Disagreements']} |\n"
    
    summary_text += """
    Conversation Flow
    """

    # Add conversation segments
    if segment_data:
        summary_text += "\n### Conversation Segments\n"
        for segment in segment_data:
            start = segment['start_time'].strftime('%H:%M:%S')
            end = segment['end_time'].strftime('%H:%M:%S')
            duration = str(segment['duration']).split('.')[0]
            
            summary_text += f"- **Segment {segment['segment_id']}:** {start} to {end} ({duration}, {segment['message_count']} messages)\n"
            summary_text += f"  - Primary participant: {segment['primary_user']}\n"
            
            # Add user distribution for segments with multiple users
            if len(segment['user_distribution']) > 1:
                summary_text += "  - User activity: "
                for user, count in segment['user_distribution'].items():
                    summary_text += f"{user} ({count}), "
                summary_text = summary_text.rstrip(", ") + "\n"
    
    # Add topic shifts
    if topic_shift_data:
        summary_text += "\n### Topic Shifts\n"
        for shift in topic_shift_data:
            time = shift['timestamp'].strftime('%H:%M:%S')
            prev_topics = ", ".join(shift['previous_topics']) if shift['previous_topics'] else "N/A"
            new_topics = ", ".join(shift['new_topics']) if shift['new_topics'] else "N/A"
            
            summary_text += f"- **Shift {shift['shift_id']}** at {time} by {shift['user']}\n"
            summary_text += f"  - From: {prev_topics}\n"
            summary_text += f"  - To: {new_topics}\n"
            summary_text += f"  - Message: \"{shift['message_text']}\"\n"
    
    # Add sentiment analysis if available
    if sentiment_data:
        summary_text += "\n### Sentiment Analysis\n"
        summary_text += f"- **Overall Sentiment:** {sentiment_data['overall_mean']:.2f} ({sentiment_data['sentiment_trend']} trend)\n"
        summary_text += f"- **Peak Positive:** {sentiment_data['max_sentiment']:.2f} by {sentiment_data['max_sentiment_user']} at {sentiment_data['max_sentiment_time'].strftime('%H:%M:%S')}\n"
        summary_text += f"- **Peak Negative:** {sentiment_data['min_sentiment']:.2f} by {sentiment_data['min_sentiment_user']} at {sentiment_data['min_sentiment_time'].strftime('%H:%M:%S')}\n"
        summary_text += f"- **Emotional Stability:** {'Stable' if sentiment_data['sentiment_stability'] < 0.3 else 'Variable'} (std={sentiment_data['sentiment_stability']:.2f})\n"
        
        # Add per-user sentiment
        summary_text += "\n**User Sentiment Profiles:**\n"
        for user, stats in sentiment_data['user_sentiments'].items():
            summary_text += f"- **{user}:** {stats['mean']:.2f} avg (range: {stats['min']:.2f} to {stats['max']:.2f})\n"
    
    # Add interaction patterns
    summary_text += "\n### Interaction Patterns\n"
    summary_text += f"- **Questions:** {interaction_data['questions']} total\n"
    summary_text += f"- **Agreements:** {interaction_data['agreements']} instances\n"
    summary_text += f"- **Disagreements:** {interaction_data['disagreements']} instances\n"
    summary_text += f"- **Emotional Peaks:** {interaction_data['emotional_peaks']} instances\n"
    
    # Add activity patterns
    if frequency_data:
        summary_text += "\n### Activity Patterns\n"
        summary_text += f"- **Average Activity:** {frequency_data['avg_messages_per_bin']:.1f} messages per {frequency_data['bin_size']}\n"
        summary_text += f"- **Peak Activity:** {frequency_data['max_messages_per_bin']} messages"
        
        if frequency_data['max_messages_time']:
            summary_text += f" at {frequency_data['max_messages_time'].strftime('%Y-%m-%d %H:%M')}\n"
        else:
            summary_text += "\n"
            
        # Add most active periods
        if frequency_data['most_active_periods']:
            summary_text += "\n**Most Active Periods:**\n"
            for period in frequency_data['most_active_periods']:
                start = period['start_time'].strftime('%H:%M')
                end = period['end_time'].strftime('%H:%M')
                summary_text += f"- {start} to {end}: {period['message_count']} messages\n"
    
    # Return the figure, metrics and text summary
    return fig, metrics, summary_text
