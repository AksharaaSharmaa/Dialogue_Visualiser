def create_topic_trees(df):
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np
    from collections import defaultdict
    import pandas as pd
    import colorsys
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    import plotly.express as px
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import traceback
    
    import nltk
    import os
    
    # Ensure nltk_data directory exists
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    # Use local nltk_data path
    nltk.data.path.append(nltk_data_path)
    
    # Download necessary NLTK resources if not already available
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }
    
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, download_dir=nltk_data_path)

    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Text preprocessing function
    def preprocess_text(text):
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        
        return " ".join(cleaned_tokens)
    
    # Ensure required columns exist
    required_columns = ['text', 'user', 'timestamp']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}. DataFrame must contain 'text', 'user', and 'timestamp'.")
    
    try:
        # Copy dataframe and sort by timestamp
        df_topic = df.copy()
        if 'timestamp' in df_topic.columns:
            df_topic = df_topic.sort_values('timestamp')
        
        # Preprocess text data
        print("Preprocessing text data...")
        df_topic['processed_text'] = df_topic['text'].apply(preprocess_text)
        
        # Filter out empty texts
        df_topic = df_topic[df_topic['processed_text'].str.strip() != ""]
        
        # Create chronological index for plotting
        df_topic['chrono_idx'] = range(len(df_topic))
        
        # Extract topics using NMF
        print("Extracting topics...")
        
        # Define min_df based on corpus size
        min_df = max(2, int(len(df_topic) * 0.01))  # At least 1% of documents or 2 documents
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=min_df,
            max_df=0.9,
            stop_words='english'
        )
        
        # Create document-term matrix
        X = vectorizer.fit_transform(df_topic['processed_text'])
        
        # Check if we have enough text data
        if X.shape[0] < 5 or X.shape[1] < 5:
            raise ValueError(f"Not enough text data: {X.shape[0]} documents with {X.shape[1]} features.")
        
        # Determine optimal number of topics
        n_topics = min(15, max(3, int(np.sqrt(len(df_topic)) / 2)))
        
        # Extract topics using NMF
        nmf_model = NMF(n_components=n_topics, random_state=42)
        nmf_topic_document_matrix = nmf_model.fit_transform(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top terms for each topic
        n_top_words = 10
        topic_terms = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_terms.append(top_features)
        
        # Assign main topic to each document
        df_topic['topic_id'] = np.argmax(nmf_topic_document_matrix, axis=1)
        
        # Calculate topic weights for each document (for mixed topic visualization)
        topic_weights = nmf_topic_document_matrix / np.sum(nmf_topic_document_matrix, axis=1, keepdims=True)
        
        # Add topic distribution to dataframe
        for i in range(n_topics):
            df_topic[f'topic_weight_{i}'] = topic_weights[:, i]
        
        # Calculate topic prominence over time
        window_size = max(5, len(df_topic) // 20)  # Dynamic window size
        topic_time_data = []
        
        for i in range(0, len(df_topic), window_size//2):  # 50% overlap
            end_idx = min(i + window_size, len(df_topic))
            if end_idx - i < window_size // 2:  # Skip small final window
                continue
                
            window_df = df_topic.iloc[i:end_idx]
            
            if 'timestamp' in window_df.columns:
                mid_time = window_df['timestamp'].mean()
                if isinstance(mid_time, pd.Timestamp):
                    time_label = mid_time.strftime('%Y-%m-%d %H:%M')
                else:
                    time_label = f"Window {i//window_size}"
            else:
                time_label = f"Window {i//window_size}"
                
            for topic_id in range(n_topics):
                topic_strength = window_df[f'topic_weight_{topic_id}'].mean()
                topic_time_data.append({
                    'window_start': i,
                    'window_end': end_idx,
                    'time_label': time_label,
                    'topic_id': topic_id,
                    'topic_strength': topic_strength,
                    'top_terms': ', '.join(topic_terms[topic_id][:5])
                })
        
        topic_time_df = pd.DataFrame(topic_time_data)
        
        # Create topic names based on top terms
        topic_names = [f"Topic {i}: {', '.join(terms[:3])}" for i, terms in enumerate(topic_terms)]
        
        # Track topic transitions
        topic_transitions = defaultdict(int)
        topic_sequence = df_topic['topic_id'].tolist()
        
        for i in range(1, len(topic_sequence)):
            source = topic_sequence[i-1]
            target = topic_sequence[i]
            transition = (source, target)
            topic_transitions[transition] += 1
        
        # Create topic tree/flow graph
        G_topic = nx.DiGraph()
        
        # Add nodes (topics)
        for topic_id in range(n_topics):
            # Count messages in this topic
            count = (df_topic['topic_id'] == topic_id).sum()
            # Get top terms for label
            label = f"Topic {topic_id}: {', '.join(topic_terms[topic_id][:3])}"
            G_topic.add_node(topic_id, 
                          size=count, 
                          label=label, 
                          terms=', '.join(topic_terms[topic_id][:7]))
        
        # Add edges (transitions)
        for (source, target), weight in topic_transitions.items():
            if source != target:  # Skip self-loops for clarity
                G_topic.add_edge(source, target, weight=weight)
        
        # Create layout - avoid using pygraphviz
        pos = nx.spring_layout(G_topic, seed=42)
        
        # Calculate node sizes based on topic prevalence
        max_size = 50
        min_size = 15
        topic_counts = df_topic['topic_id'].value_counts().to_dict()
        max_count = max(topic_counts.values()) if topic_counts else 1
        
        node_sizes = []
        for node in G_topic.nodes():
            count = topic_counts.get(node, 0)
            size = min_size + (max_size - min_size) * (count / max_count)
            node_sizes.append(size)
        
        # Topic colors (distinct colors for topics)
        topic_colors = px.colors.qualitative.Plotly[:n_topics]
        if len(topic_colors) < n_topics:
            # Generate additional colors if needed
            additional_colors = []
            for i in range(n_topics - len(topic_colors)):
                h = i / (n_topics - len(topic_colors))
                r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, 0.8, 0.8)]
                additional_colors.append(f'rgb({r},{g},{b})')
            topic_colors.extend(additional_colors)
        
        node_colors = [topic_colors[node % len(topic_colors)] for node in G_topic.nodes()]
        
        # Create node traces
        node_x, node_y = [], []
        for node in G_topic.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G_topic.nodes[n]['label'] for n in G_topic.nodes()],
            textposition="top center",
            hovertext=[f"Topic {n}: {G_topic.nodes[n]['terms']}<br>Messages: {G_topic.nodes[n]['size']}" for n in G_topic.nodes()],
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white')
            )
        )
        
        # Create edge traces
        edge_traces = []
        
        # Define edge width scaling
        max_edge_width = 5
        min_edge_width = 1
        max_weight = max([d['weight'] for _, _, d in G_topic.edges(data=True)]) if G_topic.edges(data=True) else 1
        
        # Create edges with varying thickness and arrows
        for edge in G_topic.edges(data=True):
            source, target, data = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            weight = data.get('weight', 1)
            width = min_edge_width + (max_edge_width - min_edge_width) * (weight / max_weight)
            
            # Calculate curved paths for edges
            curve_factor = 0.2
            
            # Check if there's a reverse edge
            has_reverse = G_topic.has_edge(target, source)
            if has_reverse:
                # Increase curve for bidirectional edges
                curve_factor = 0.4
            
            # Calculate control point for curved edge
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            # Normal vector to edge
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            nx = -dy / length if length > 0 else 0
            ny = dx / length if length > 0 else 0
            
            # Control point position
            cx = mid_x + curve_factor * nx * length
            cy = mid_y + curve_factor * ny * length
            
            # Generate quadratic Bezier curve
            t = np.linspace(0, 1, 30)
            x = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
            y = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
            
            # Color based on source topic
            color = topic_colors[source % len(topic_colors)]
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=x, y=y,
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=f"Topic {source} → Topic {target}: {weight} transitions",
                mode='lines',
                opacity=0.7,
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
            # Add arrow
            arrow_pos = 0.9  # Position arrow near end
            arrow_idx = int(arrow_pos * (len(x) - 1))
            
            arrow_trace = go.Scatter(
                x=[x[arrow_idx]],
                y=[y[arrow_idx]],
                mode='markers',
                marker=dict(
                    symbol='arrow',
                    size=10,
                    color=color,
                    angle=np.degrees(np.arctan2(
                        y[arrow_idx+1] - y[arrow_idx-1],
                        x[arrow_idx+1] - x[arrow_idx-1]
                    ))
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(arrow_trace)
        
        # Create topic evolution over time visualization
        topic_evolution_data = []
        
        # Group by windows and calculate topic prominence
        grouped_time_df = topic_time_df.groupby(['time_label', 'topic_id'])
        for (time_label, topic_id), group in grouped_time_df:
            topic_evolution_data.append({
                'time': time_label,
                'topic': topic_names[topic_id],
                'strength': group['topic_strength'].mean(),
                'top_terms': group['top_terms'].iloc[0]
            })
        
        topic_evolution_df = pd.DataFrame(topic_evolution_data)
        
        # Create heatmap of topic evolution
        topic_heatmap_df = topic_evolution_df.pivot(
            index='topic', 
            columns='time', 
            values='strength'
        ).fillna(0)
        
        # Sort topics by overall prevalence
        topic_totals = topic_heatmap_df.sum(axis=1)
        topic_heatmap_df = topic_heatmap_df.loc[topic_totals.sort_values(ascending=False).index]
        
        # Create radial layout for topic relationships
        def generate_radial_coords(n, radius=1):
            coords = []
            for i in range(n):
                angle = 2 * np.pi * i / n
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                coords.append((x, y))
            return coords
        
        # Calculate connection strength between topics
        topic_similarity = np.zeros((n_topics, n_topics))
        
        # Use transitions as a measure of topic similarity
        for (source, target), weight in topic_transitions.items():
            topic_similarity[source, target] += weight
        
        # Add cosine similarity between topic vectors
        from sklearn.metrics.pairwise import cosine_similarity
        topic_vectors = nmf_model.components_
        cosine_sim = cosine_similarity(topic_vectors)
        
        # Normalize and combine metrics
        if np.max(topic_similarity) > 0:
            topic_similarity = topic_similarity / np.max(topic_similarity)
        topic_similarity = 0.7 * topic_similarity + 0.3 * cosine_sim
        
        # Create radial layout
        radial_coords = generate_radial_coords(n_topics, radius=10)
        
        # Generate insights about topic patterns
        insights = []
        
        # Top topics
        topic_frequency = df_topic['topic_id'].value_counts()
        top_topics = topic_frequency.nlargest(3).index.tolist()
        top_topic_terms = [', '.join(topic_terms[t][:5]) for t in top_topics]
        
        insights.append(f"The conversation is dominated by topics around {top_topic_terms[0]}, representing {topic_frequency[top_topics[0]]/len(df_topic)*100:.1f}% of messages.")
        
        # Topic transitions
        common_transitions = []
        for (source, target), count in sorted(topic_transitions.items(), key=lambda x: x[1], reverse=True)[:3]:
            if source != target:
                source_terms = ', '.join(topic_terms[source][:3])
                target_terms = ', '.join(topic_terms[target][:3])
                common_transitions.append(f"{source_terms} → {target_terms}")
        
        if common_transitions:
            insights.append(f"Common topic shifts include: {'; '.join(common_transitions)}.")
        
        # Topic evolution
        emerging_topics = []
        for topic_id in range(n_topics):
            if topic_id in topic_time_df['topic_id'].values:
                topic_data = topic_time_df[topic_time_df['topic_id'] == topic_id]
                if len(topic_data) >= 3:
                    # Check if trend is increasing
                    first_half = topic_data.iloc[:len(topic_data)//2]['topic_strength'].mean()
                    second_half = topic_data.iloc[len(topic_data)//2:]['topic_strength'].mean()
                    
                    if second_half > first_half * 1.5:  # 50% increase
                        emerging_topics.append(f"{', '.join(topic_terms[topic_id][:3])}")
        
        if emerging_topics:
            insights.append(f"Emerging topics include: {', '.join(emerging_topics)}.")
        
        # User engagement
        if 'user' in df_topic.columns:
            user_topic_counts = df_topic.groupby(['user', 'topic_id']).size().unstack(fill_value=0)
            most_active_users = df_topic['user'].value_counts().nlargest(3)
            
            if not most_active_users.empty:
                user_insights = []
                for user, count in most_active_users.items():
                    if user in user_topic_counts.index:
                        user_topics = user_topic_counts.loc[user]
                        main_topic = user_topics.idxmax()
                        user_insights.append(f"{user} ({count} messages, mainly on {', '.join(topic_terms[main_topic][:3])})")
                
                if user_insights:
                    insights.append(f"Most active participants: {'; '.join(user_insights)}.")
        
        # Add topic descriptions to insights instead of visualization
        topic_descriptions = []
        for i in range(n_topics):
            topic_descriptions.append(f"Topic {i}: {', '.join(topic_terms[i][:7])}")
        
        insights.append("Topic Descriptions:<br>" + "<br>".join(topic_descriptions))
        
        # Join insights
        all_insights = "<br>".join(insights)
        
        # Create a single consolidated figure
        from plotly.subplots import make_subplots
        
        # Create consolidated visualization with 3 components - improved layout and spacing
        consolidated_fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                  [{"type": "heatmap", "colspan": 2}, {}]],
            subplot_titles=("Topic Flow Diagram", "Topic Relationships", 
                           "Topic Evolution Over Time"),
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
            row_heights=[0.5, 0.5]
        )
        
        # 1. Add topic flow diagram (top left)
        for trace in edge_traces:
            consolidated_fig.add_trace(trace, row=1, col=1)
        consolidated_fig.add_trace(node_trace, row=1, col=1)
        
        # 2. Add topic relationships (top right)
        # Create traces for radial visualization
        radial_nodes_trace = go.Scatter(
            x=[coord[0] for coord in radial_coords],
            y=[coord[1] for coord in radial_coords],
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=[topic_names[i] for i in range(n_topics)],
            textposition="middle center",
            hoverinfo='text',
            hovertext=[f"{topic_names[i]}<br>Top terms: {', '.join(topic_terms[i][:5])}" for i in range(n_topics)]
        )
        
        # Create edges between related topics
        radial_edges = []
        
        for i in range(n_topics):
            for j in range(i+1, n_topics):
                similarity = topic_similarity[i, j]
                
                # Only draw edges for significant relationships
                if similarity > 0.1:
                    x0, y0 = radial_coords[i]
                    x1, y1 = radial_coords[j]
                    
                    # Width based on similarity
                    width = similarity * 3
                    
                    # Opacity based on similarity
                    opacity = min(0.8, similarity)
                    
                    edge_trace = go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(width=width, color='rgba(100,100,100,'+str(opacity)+')'),
                        hoverinfo='text',
                        hovertext=f"Similarity: {similarity:.2f}"
                    )
                    consolidated_fig.add_trace(edge_trace, row=1, col=2)
        
        consolidated_fig.add_trace(radial_nodes_trace, row=1, col=2)
        
        # 3. Add topic evolution heatmap (bottom) with improved spacing
        consolidated_fig.add_trace(
            go.Heatmap(
                z=topic_heatmap_df.values,
                x=topic_heatmap_df.columns,
                y=topic_heatmap_df.index,
                colorscale='Blues',
                hoverongaps=False,
                hovertemplate='Time: %{x}<br>%{y}<br>Strength: %{z:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout with better spacing
        consolidated_fig.update_layout(
            height=1200,
            width=2000,
            title_text="Topic Analysis Dashboard",
            showlegend=False,
            margin=dict(l=60, r=50, t=100, b=100)
        )
        
        # Update axes properties
        consolidated_fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        consolidated_fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        consolidated_fig.update_xaxes(title_text="Time", row=2, col=1)
        
        consolidated_fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        consolidated_fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        consolidated_fig.update_yaxes(title_text="Topics", row=2, col=1)
        
        # REMOVED THE INSIGHTS ANNOTATION FROM THE VISUALIZATION
        # Instead, we'll return the insights separately
        
        # Return consolidated figure and data
        return {
            'visualization': consolidated_fig,
            'topic_data': df_topic,
            'topic_terms': topic_terms,
            'topic_names': topic_names,
            'insights': insights,  # Return insights as a separate item
            'all_insights_html': all_insights  # Return HTML formatted insights
        }
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in topic extraction: {e}")
        print(f"Traceback: {error_traceback}")
        
        # Return empty visualization
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Not enough data for topic analysis",
            annotations=[
                dict(
                    text=f"Error: {str(e)}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ]
        )
        return {
            'visualization': empty_fig, 
            'topic_data': df,
            'topic_terms': [],
            'topic_names': [],
            'insights': [f"Error in topic analysis: {str(e)}"],
            'all_insights_html': f"Error in topic analysis: {str(e)}"
        }
