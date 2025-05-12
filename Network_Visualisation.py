def create_network_visualization(df):
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np
    from collections import defaultdict
    import pandas as pd
    import colorsys
    import random
    from community import community_louvain  # pip install python-louvain
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import google.generativeai as genai
    import os
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track user statistics
    user_message_count = defaultdict(int)
    user_sentiment = defaultdict(list)
    user_responses = defaultdict(int)  # Count of responses received
    interaction_counts = defaultdict(int)  # Count interactions between users
    user_text = defaultdict(list)  # Store text for content analysis
    
    # Add nodes (users)
    all_users = set(df['user'].unique())
    
    # Add additional users from reply_to if they're not in the user column
    if 'reply_to' in df.columns:
        for reply_id in df['reply_to'].dropna().unique():
            # Find the user who wrote the message with this ID
            reply_rows = df[df['id'] == reply_id]
            if not reply_rows.empty:
                all_users.add(reply_rows.iloc[0]['user'])
    
    # Initialize graph with all users
    for user in all_users:
        G.add_node(user)
        
    # Count messages per user and calculate average sentiment
    for _, row in df.iterrows():
        user = row['user']
        user_message_count[user] += 1
        
        # Store text for content analysis
        if 'text' in row and pd.notna(row['text']):
            user_text[user].append(row['text'])
            
        # Add sentiment if available
        if 'sentiment_compound' in row:
            user_sentiment[user].append(row['sentiment_compound'])
    
    # Add edges (interactions)
    if 'reply_to' in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row['reply_to']):
                # Find the user who wrote the message being replied to
                reply_rows = df[df['id'] == row['reply_to']]
                
                if not reply_rows.empty:
                    target_user = reply_rows.iloc[0]['user']
                    source_user = row['user']
                    
                    # Track the interaction (even for self-loops)
                    interaction_key = (source_user, target_user)
                    interaction_counts[interaction_key] += 1
                    
                    # Track responses received
                    user_responses[target_user] += 1
    
    # Add weighted edges to the graph
    for (source, target), weight in interaction_counts.items():
        # Add the edge with weight attribute
        G.add_edge(source, target, weight=weight)
    
    # Create an undirected copy for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities using Louvain method
    partition = community_louvain.best_partition(G_undirected)
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    # Calculate network metrics
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight')
    clustering = nx.clustering(G_undirected)
    
    # Calculate average sentiment per user
    avg_sentiment = {}
    for user, sentiments in user_sentiment.items():
        if sentiments:
            avg_sentiment[user] = np.mean(sentiments)
        else:
            avg_sentiment[user] = 0
    
    # Content-based subgroup detection
    # Combine all text for each user
    user_combined_text = {user: " ".join(texts) for user, texts in user_text.items() if texts}
    
    # Only perform text analysis if we have sufficient data
    content_clusters = {}
    if len(user_combined_text) >= 3:
        try:
            # Create feature vectors using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', 
                                         min_df=2, max_df=0.8)
            users = list(user_combined_text.keys())
            texts = [user_combined_text[user] for user in users]
            
            # Only proceed if we have enough text data
            if len(texts) >= 3 and sum(len(t) > 50 for t in texts) >= 3:
                X = vectorizer.fit_transform(texts)
                
                # Determine optimal number of clusters (2-5)
                k = min(max(2, len(texts) // 3), 5)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                # Map users to content clusters
                for i, user in enumerate(users):
                    content_clusters[user] = int(clusters[i])
        except Exception as e:
            print(f"Content clustering error: {str(e)}")
    
    # Generate network layout based on community structure
    if len(G.nodes) > 50:
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Create edge traces with improved visualization
    edge_traces = []
    
    # Define edge width scaling
    max_edge_width = 8
    min_edge_width = 1
    max_weight = max(interaction_counts.values()) if interaction_counts else 1
    
    # Track edges for curved drawing
    edge_count = defaultdict(int)
    
    # Create edges with varying thickness and curvature
    for edge in G.edges(data=True):
        source, target, data = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        weight = data.get('weight', 1)
        width = min_edge_width + (max_edge_width - min_edge_width) * (weight / max_weight)
        
        # Calculate sentiment of this interaction if available
        interaction_sentiment = 0
        interaction_rows = df[(df['user'] == source) & (df['reply_to'].isin(df[df['user'] == target]['id']))]
        if 'sentiment_compound' in df.columns and not interaction_rows.empty:
            interaction_sentiment = interaction_rows['sentiment_compound'].mean()
        
        # Color based on sentiment with improved visibility
        if interaction_sentiment < -0.05:
            color = f'rgba(255, 0, 0, {min(0.9, 0.4 + abs(interaction_sentiment))})'
        elif interaction_sentiment > 0.05:
            color = f'rgba(0, 0, 255, {min(0.9, 0.4 + interaction_sentiment)})'
        else:
            color = 'rgba(150, 150, 150, 0.7)'
        
        # Add curvature to distinguish bidirectional edges
        edge_pair = tuple(sorted([source, target]))
        edge_count[edge_pair] += 1
        curve = 0.2 if edge_count[edge_pair] > 1 or source == target else 0
        
        if source == target:  # Self-loop
            # Create loop
            loop_size = 0.15
            ctrl1_x = x0 + loop_size
            ctrl1_y = y0
            ctrl2_x = x0
            ctrl2_y = y0 + loop_size
            
            # Bezier curve for self-loop
            t = np.linspace(0, 1, 50)
            x = (1-t)**2 * x0 + 2*(1-t)*t * ctrl1_x + t**2 * x0
            y = (1-t)**2 * y0 + 2*(1-t)*t * ctrl2_y + t**2 * y0
            
            edge_trace = go.Scatter(
                x=x, y=y,
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=f'{source} → {source}: {weight} self-references<br>Sentiment: {interaction_sentiment:.2f}',
                mode='lines',
                showlegend=False
            )
        else:
            # Calculate curve for regular edges
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            # Perpendicular offset for curve
            dx = x1 - x0
            dy = y1 - y0
            offset = curve * np.sqrt(dx**2 + dy**2)
            
            # Normal vector for curve (perpendicular to edge)
            nx = -dy / np.sqrt(dx**2 + dy**2) if dx**2 + dy**2 > 0 else 0
            ny = dx / np.sqrt(dx**2 + dy**2) if dx**2 + dy**2 > 0 else 0
            
            # Control point for curved edge
            cx = mid_x + offset * nx
            cy = mid_y + offset * ny
            
            # Generate points along quadratic Bezier curve
            t = np.linspace(0, 1, 30)
            bezier_x = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
            bezier_y = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
            
            # Add arrow to indicate direction
            arrow_pos = 0.9  # Position arrow near the end
            arrow_idx = int(arrow_pos * (len(bezier_x) - 1))
            
            edge_trace = go.Scatter(
                x=bezier_x, y=bezier_y,
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=f'{source} → {target}: {weight} interactions<br>Sentiment: {interaction_sentiment:.2f}',
                mode='lines',
                showlegend=False
            )
            
            # Add arrow marker
            arrow_trace = go.Scatter(
                x=[bezier_x[arrow_idx]],
                y=[bezier_y[arrow_idx]],
                mode='markers',
                marker=dict(
                    symbol='arrow', 
                    size=15,
                    color=color,
                    angle=np.arctan2(
                        bezier_y[arrow_idx+1] - bezier_y[arrow_idx-1],
                        bezier_x[arrow_idx+1] - bezier_x[arrow_idx-1]
                    ) * 180 / np.pi
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(arrow_trace)
            
        edge_traces.append(edge_trace)
    
    # Create node trace with improved visualization
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Scale node sizes based on message count
    max_messages = max(user_message_count.values()) if user_message_count else 1
    min_size = 15
    max_size = 50
    node_sizes = []
    for node in G.nodes():
        size = min_size + (max_size - min_size) * (user_message_count.get(node, 1) / max_messages)
        node_sizes.append(size)
    
    # Generate node colors based on community detection
    community_colors = {}
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    for i, community_id in enumerate(communities.keys()):
        community_colors[community_id] = distinct_colors[i % len(distinct_colors)]
    
    node_colors = [community_colors[partition[node]] for node in G.nodes()]
    
    # Create hover text with user stats
    node_text = []
    for node in G.nodes():
        sentiment_avg = avg_sentiment.get(node, 0)
        sentiment_text = "Positive" if sentiment_avg > 0.05 else "Negative" if sentiment_avg < -0.05 else "Neutral"
        
        community_id = partition[node]
        community_size = len(communities[community_id])
        
        # Get content topic if available
        content_topic = f"Content Group: {content_clusters.get(node, 'Unknown')}" if node in content_clusters else ""
        
        text = (f"User: {node}<br>"
                f"Messages: {user_message_count.get(node, 0)}<br>"
                f"Responses received: {user_responses.get(node, 0)}<br>"
                f"Sentiment: {sentiment_avg:.2f} ({sentiment_text})<br>"
                f"Community: {community_id} (size: {community_size})<br>"
                f"Centrality: {betweenness.get(node, 0):.3f}<br>"
                f"Influence: {eigenvector.get(node, 0):.3f}<br>"
                f"{content_topic}")
        node_text.append(text)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n if len(n) < 15 else n[:12]+'...' for n in G.nodes()],  # Display usernames
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale=[[0, color] for i, color in enumerate(node_colors)],
            color=list(range(len(node_colors))),
            size=node_sizes,
            line=dict(width=2, color='rgba(255, 255, 255, 0.8)'),

        )
    )
    
    # Add community hulls (convex hulls around each community)
    hull_traces = []
    for community_id, members in communities.items():
        if len(members) < 2:
            continue  # Skip single-member communities
            
        color = community_colors[community_id]
        
        # Get positions of community members
        comm_pos = np.array([pos[member] for member in members])
        
        # Try to compute convex hull
        if len(comm_pos) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(comm_pos)
                
                # Get hull vertices in order
                vertices = comm_pos[hull.vertices]
                
                # Add first point to close the polygon
                vertices = np.vstack([vertices, vertices[0]])
                
                # Create a shape for the community
                hull_trace = go.Scatter(
                    x=vertices[:, 0], 
                    y=vertices[:, 1],
                    fill='toself',
                    fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                    line=dict(color=color, width=1),
                    hoverinfo='text',
                    text=f"Community {community_id}: {len(members)} members",
                    mode='lines',
                    showlegend=False,
                    name=f"Community {community_id}"
                )
                hull_traces.append(hull_trace)
            except Exception as e:
                print(f"Error creating hull for community {community_id}: {str(e)}")
    
    # Generate network analysis and statistics
    network_stats = {
        "node_count": len(G.nodes),
        "edge_count": len(G.edges),
        
        "communities": len(communities),
        "largest_community": max([len(c) for c in communities.values()]),
        "modularity": community_louvain.modularity(partition, G_undirected),
        "central_users": sorted([(u, score) for u, score in betweenness.items()], 
                               key=lambda x: x[1], reverse=True)[:5],
        "influential_users": sorted([(u, score) for u, score in eigenvector.items()], 
                                  key=lambda x: x[1], reverse=True)[:5],
        "active_users": sorted([(u, count) for u, count in user_message_count.items()], 
                              key=lambda x: x[1], reverse=True)[:5]
    }
    
    # Analyze community interactions
    community_interactions = defaultdict(int)
    cross_community = 0
    within_community = 0
    
    for (source, target), weight in interaction_counts.items():
        source_comm = partition.get(source)
        target_comm = partition.get(target)
        
        if source_comm == target_comm:
            within_community += weight
        else:
            cross_community += weight
            community_interactions[(source_comm, target_comm)] += weight
    
    # Calculate echo chamber metrics
    echo_chamber_index = within_community / (within_community + cross_community) if (within_community + cross_community) > 0 else 0
    network_stats["echo_chamber_index"] = echo_chamber_index
    
    # Find most active community
    community_activity = defaultdict(int)
    for user, count in user_message_count.items():
        if user in partition:
            community_activity[partition[user]] += count
    
    # Find bridges between communities
    bridges = []
    for u, v in G.edges():
        if partition[u] != partition[v]:
            bridges.append((u, v))
    
    network_stats["bridges"] = len(bridges)
    
    # Use Gemini to generate insights
    try:
        # Configure Gemini API
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty in secrets")
    except KeyError:
        raise ValueError("GEMINI_API_KEY is not set in Streamlit secrets")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare data for Gemini
        network_summary = f"""
        Network Analysis:
        - Total users: {len(G.nodes)}
        - Total interactions: {len(G.edges)}
        - Number of communities: {len(communities)}
        - Echo chamber index: {echo_chamber_index:.2f} (higher means more within-community interaction)
        - Number of cross-community connections: {len(bridges)}
        
        Top users by centrality: {', '.join([u[0] for u in network_stats["central_users"][:3]])}
        Top users by influence: {', '.join([u[0] for u in network_stats["influential_users"][:3]])}
        Most active users: {', '.join([u[0] for u in network_stats["active_users"][:3]])}
        
        Community sizes: {', '.join([f"Community {comm_id}: {len(members)}" for comm_id, members in communities.items()])}
        
        Interaction patterns:
        - Within-community interactions: {within_community}
        - Cross-community interactions: {cross_community}
        """
        
        # Add sentiment analysis if available
        if user_sentiment:
            positive_users = sum(1 for s in avg_sentiment.values() if s > 0.05)
            negative_users = sum(1 for s in avg_sentiment.values() if s < -0.05)
            neutral_users = len(avg_sentiment) - positive_users - negative_users
            
            network_summary += f"""
            Sentiment analysis:
            - Positive users: {positive_users} ({positive_users/len(avg_sentiment)*100:.1f}%)
            - Negative users: {negative_users} ({negative_users/len(avg_sentiment)*100:.1f}%)
            - Neutral users: {neutral_users} ({neutral_users/len(avg_sentiment)*100:.1f}%)
            """
        
        # Generate prompt for Gemini
        prompt = f"""
        You are a network analysis expert. Based on the following social network data, provide 5-7 key insights about the network structure, 
        community dynamics, influence patterns, and potential echo chambers. Focus on what these patterns reveal about communication flow 
        and group dynamics. Keep each insight concise (1-2 sentences). Also print the above sentiment analysis in points and all the network summary points such as 
        total users, total interactions everything in points. Also explain the meaning of terms used
        such as centrality, influence, echo chamber index etc. Do not make bullets, just start with a new
        line after each point.
        
        {network_summary}
        
        Format your response as a list of insights without numbering, introductions or explanations.
        """
        
        # Generate insights with Gemini
        response = model.generate_content(prompt)
        gemini_insights = response.text.strip().split('\n')
        
        # Clean up insights
        insights = [insight.strip('• -') for insight in gemini_insights if insight.strip()]
        
    except Exception as e:
        print(f"Error using Gemini API: {str(e)}")
        # Fallback insights if Gemini fails
        insights = []
        
        # General network structure
        insights.append(f"Network has {len(G.nodes)} users in {len(communities)} distinct communities.")
        
        # Echo chamber analysis
        if echo_chamber_index > 0.7:
            insights.append(f"Strong echo chamber effect detected (index: {echo_chamber_index:.2f}). "
                        f"Users primarily interact within their own communities.")
        elif echo_chamber_index < 0.3:
            insights.append(f"Low echo chamber effect (index: {echo_chamber_index:.2f}). "
                        f"Significant cross-community dialogue.")
        
        # Key influencers
        if network_stats["influential_users"]:
            top_influencer = network_stats["influential_users"][0]
            insights.append(f"User '{top_influencer[0]}' is the most influential participant, "
                        f"receiving responses from diverse users.")
        
        # Community dynamics
        most_active_comm = max(community_activity.items(), key=lambda x: x[1]) if community_activity else (None, 0)
        if most_active_comm[0] is not None:
            insights.append(f"Community {most_active_comm[0]} is the most active with {most_active_comm[1]} messages.")
        
        # Find main bridge users (between communities)
        if bridges:
            bridge_users = defaultdict(int)
            for u, v in bridges:
                bridge_users[u] += 1
                bridge_users[v] += 1
            
            top_bridges = sorted(bridge_users.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_bridges:
                insights.append(f"Users {', '.join([u[0] for u in top_bridges])} function as bridges "
                            f"connecting different community groups.")

    # Create the figure
    fig = go.Figure(
        data=hull_traces + edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1000,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"marker.size": node_sizes}],
                            label="Size by Message Count",
                            method="update"
                        ),
                        dict(
                            args=[{"marker.size": [15 + 35 * betweenness.get(node, 0) for node in G.nodes()]}],
                            label="Size by Centrality",
                            method="update"
                        ),
                        dict(
                            args=[{"marker.size": [15 + 35 * eigenvector.get(node, 0) for node in G.nodes()]}],
                            label="Size by Influence",
                            method="update"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
    )

    # Add legend for edge colors (inside visualization)
    fig.add_annotation(
        x=1.0,
        y=1.0,
        xref="paper",
        yref="paper",
        text="Edge Colors:<br>🔴 Negative Sentiment<br>🔵 Positive Sentiment<br>⚪ Neutral",
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4
    )

    '''# Add community information box (inside visualization)
    community_info = "<br>".join([f"Community {comm_id}: {len(members)} members" 
                                for comm_id, members in communities.items()])
    fig.add_annotation(
        x=0.01,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"<b>Communities:</b><br>{community_info}",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4
    )'''

    # Adjust layout for visualization without insights inside
    fig.update_layout(
        height=800,  # Reduced height since we're not including insights in visualization
        margin=dict(b=150, l=50, r=50, t=50)
    )

    return fig, insights
