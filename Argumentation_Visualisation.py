def create_argumentation_graph(df):
    import pandas as pd
    import plotly.graph_objects as go
    import networkx as nx
    import re
    import nltk
    from nltk.tokenize import sent_tokenize
    from collections import defaultdict, Counter
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    import spacy
    import random
    import en_core_web_sm

    # Direct loading from the imported package
    nlp = en_core_web_sm.load()
   
    # Initialize colors for different argument types
    colors = {
        'claim': '#1f77b4',       # Blue
        'counterclaim': '#d62728', # Red
        'evidence': '#2ca02c',     # Green
        'question': '#9467bd',     # Purple
        'agreement': '#17becf',    # Cyan
        'disagreement': '#e377c2', # Pink
        'clarification': '#ff7f0e' # Orange
    }
    
    # Make a copy of the data to avoid modifying the original
    data = df.copy()
    
    # Ensure required columns exist
    required_columns = ['text', 'user']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Add reply_to column if not present
    if 'reply_to' not in data.columns:
        data['reply_to'] = None
    
    # Add id column if not present
    if 'id' not in data.columns:
        data['id'] = range(1, len(data) + 1)
    
    # Function to detect argument type based on text content
    def detect_argument_type(text, prev_text=None):
        text_lower = text.lower()
        
        # Simple pattern matching for argument types
        patterns = {
            'question': [r'\?$', r'^(what|why|how|when|where|who|which)[\s\W]'],
            'agreement': [r'\b(agree|concur|exactly|right|correct|true|yes)\b', r'^(yes|yeah|indeed|absolutely)[\s\W]'],
            'disagreement': [r'\b(disagree|incorrect|wrong|false|no|but)\b', r'^(no|nope|however|actually|but)[\s\W]'],
            'evidence': [r'\b(because|since|research|study|data|evidence|according|show|demonstrate)\b'],
            'clarification': [r'\b(mean|clarify|explain|elaborate|understand)\b'],
        }
        
        # Check for patterns
        for arg_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text_lower):
                    return arg_type
        
        # Default: if replying to something, it's likely a counterclaim or evidence
        if prev_text is not None:
            # If there's disagreement language or contradictory content
            disagreement_indicators = ['but', 'however', 'though', 'disagree', 'incorrect', 'wrong']
            if any(indicator in text_lower for indicator in disagreement_indicators):
                return 'counterclaim'
            else:
                # Check if it's adding additional evidence or supporting a claim
                support_indicators = ['also', 'additionally', 'furthermore', 'moreover', 'agree']
                if any(indicator in text_lower for indicator in support_indicators):
                    return 'evidence'
        
        # Default to claim
        return 'claim'
    
    # Function to extract key phrases from text
    def extract_key_phrases(text, max_phrases=3):
        if not text or len(text) < 5:
            return [""]
        
        if nlp:
            # Use spaCy for more sophisticated extraction
            doc = nlp(text)
            
            # Extract noun phrases as potential key phrases
            noun_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 6:  # Limit to reasonable length
                    noun_phrases.append(chunk.text)
            
            # If we have enough noun phrases, return them
            if len(noun_phrases) >= max_phrases:
                return noun_phrases[:max_phrases]
            
            # Otherwise, extract important sentences
            sentences = sent_tokenize(text)
            if len(sentences) >= max_phrases:
                return [s[:50] + '...' if len(s) > 50 else s for s in sentences[:max_phrases]]
        
        # Fallback: just use the first sentence or part of it
        sentences = sent_tokenize(text)
        first_sent = sentences[0] if sentences else text
        if len(first_sent) > 60:
            return [first_sent[:60] + '...']
        return [first_sent]
    
    # Process each message to identify argument structure
    for i, row in data.iterrows():
        # Get previous message if this is a reply
        prev_text = None
        if row['reply_to'] is not None and not pd.isna(row['reply_to']):
            prev_row = data[data['id'] == row['reply_to']]
            if not prev_row.empty:
                prev_text = prev_row.iloc[0]['text']
        
        # Detect argument type
        data.at[i, 'arg_type'] = detect_argument_type(row['text'], prev_text)
        
        # Extract key phrases
        data.at[i, 'key_phrases'] = extract_key_phrases(row['text'])
    
    # Create a graph to represent the argument structure
    G = nx.DiGraph()
    
    # Add nodes for each message
    for i, row in data.iterrows():
        node_id = row['id']
        user = row['user']
        text = row['text']
        arg_type = row['arg_type']
        
        # Truncate text for display
        display_text = text[:100] + '...' if len(text) > 100 else text
        
        # Add node with attributes
        G.add_node(
            node_id, 
            user=user, 
            text=text, 
            display_text=display_text,
            arg_type=arg_type,
            key_phrases=row['key_phrases']
        )
    
    # Add edges for replies
    for i, row in data.iterrows():
        if row['reply_to'] is not None and not pd.isna(row['reply_to']):
            # Check if the target node exists
            if row['reply_to'] in G:
                G.add_edge(row['reply_to'], row['id'], type=row['arg_type'])
    
    # Create a layout for the graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Prepare node data for plotting
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Prepare hover text
        node_data = G.nodes[node]
        hover_text = f"User: {node_data['user']}<br>Type: {node_data['arg_type']}<br>Text: {node_data['display_text']}"
        node_text.append(hover_text)
        
        # Node color based on argument type
        node_color.append(colors.get(node_data['arg_type'], '#7f7f7f'))
        
        # Node size based on message length and importance
        base_size = 15
        # Adjust size based on connectivity and message length
        size_factor = 1 + 0.05 * len(G.edges(node)) + 0.01 * min(len(node_data['text']), 100)
        node_size.append(base_size * size_factor)
    
    # Create edge traces grouped by color
    edge_traces = {}
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Edge type and color
        edge_type = G.edges[edge]['type']
        color = colors.get(edge_type, '#7f7f7f')
        
        # Edge width based on argument strength/importance
        width = 2 if edge_type in ['evidence', 'counterclaim'] else 1
        
        # Hover text for the edge
        source_node = G.nodes[edge[0]]
        source_user = source_node['user']
        target_node = G.nodes[edge[1]]
        target_user = target_node['user']
        
        edge_hover = f"From: {source_user}<br>To: {target_user}<br>Type: {edge_type}"
        
        # Create or update the trace for this color
        if color not in edge_traces:
            edge_traces[color] = {
                'x': [], 'y': [], 'text': [], 'width': width
            }
        
        # Add line segments
        edge_traces[color]['x'].extend([x0, x1, None])
        edge_traces[color]['y'].extend([y0, y1, None])
        edge_traces[color]['text'].extend([edge_hover, edge_hover, None])
    
    # Create node traces
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#888'),
        )
    )
    
    # Create a separate edge trace for each color
    all_edge_traces = []
    for color, trace_data in edge_traces.items():
        all_edge_traces.append(
            go.Scatter(
                x=trace_data['x'], 
                y=trace_data['y'],
                mode='lines',
                hoverinfo='text',
                text=trace_data['text'],
                line=dict(color=color, width=trace_data['width']),
                opacity=0.7
            )
        )
    
    # Add label traces for important nodes
    # We'll label nodes that are either:
    # - A root claim (no incoming edges)
    # - Have multiple replies
    # - Are counterclaims or evidence
    labels = []
    label_x = []
    label_y = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        x, y = pos[node]
        
        # Decide if this node should be labeled
        should_label = (
            len(G.in_edges(node)) == 0 or  # Root claims
            len(G.out_edges(node)) > 1 or  # Multiple replies
            node_data['arg_type'] in ['counterclaim', 'evidence']  # Important argument types
        )
        
        if should_label:
            # Use the first key phrase or a placeholder
            if node_data['key_phrases']:
                label = node_data['key_phrases'][0]
            else:
                label = f"{node_data['arg_type'].capitalize()}"
            
            # Truncate long labels
            if len(label) > 25:
                label = label[:22] + '...'
            
            labels.append(label)
            label_x.append(x)
            label_y.append(y + 0.03)  # Slightly above the node
    
    label_trace = go.Scatter(
        x=label_x, y=label_y,
        mode='text',
        text=labels,
        textposition="top center",
        textfont=dict(size=10, color='black'),
        hoverinfo='none'
    )
    
    # Create the figure (adding all edge traces)
    fig_data = all_edge_traces + [node_trace, label_trace]
    fig = go.Figure(data=fig_data,
                    layout=go.Layout(
                        title="Argumentation Structure Graph",
                        font=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='rgba(255,255,255,1)',
                        plot_bgcolor='rgba(255,255,255,1)',
                    )
                   )
    
    # Add a legend
    legend_x = max(node_x) + 0.1
    legend_y = max(node_y)
    
    for i, (arg_type, color) in enumerate(colors.items()):
        fig.add_trace(go.Scatter(
            x=[legend_x],
            y=[legend_y - i * 0.1],
            mode='markers',
            marker=dict(size=10, color=color),
            name=arg_type.capitalize(),
            hoverinfo='name'
        ))
    
    # Analyze the argument structure
    analysis = {}
    
    # Count argument types
    arg_counts = Counter(data['arg_type'])
    analysis['arg_counts'] = arg_counts
    
    # Find most prolific arguers
    user_arg_counts = data.groupby('user')['arg_type'].count().sort_values(ascending=False)
    analysis['top_arguers'] = user_arg_counts.head(5).to_dict()
    
    # Find most persuasive users (those who receive agreements)
    if 'agreement' in data['arg_type'].values:
        agreements = data[data['arg_type'] == 'agreement']
        persuasive_users = []
        
        for _, row in agreements.iterrows():
            if row['reply_to'] is not None and not pd.isna(row['reply_to']):
                orig_message = data[data['id'] == row['reply_to']]
                if not orig_message.empty:
                    persuasive_users.append(orig_message.iloc[0]['user'])
        
        analysis['persuasive_users'] = Counter(persuasive_users)
    
    # Find most controversial claims (those with both agreements and disagreements)
    controversial = []
    for node_id in G.nodes():
        # Get all replies to this node
        replies = [G.nodes[succ]['arg_type'] for succ in G.successors(node_id)]
        
        # Check if it has both agreements and disagreements
        if 'agreement' in replies and 'disagreement' in replies:
            controversial.append(node_id)
    
    analysis['controversial_claims'] = [G.nodes[node_id]['display_text'] for node_id in controversial]
    
    # Generate insights
    insights = []
    
    # Overall argument structure
    insights.append(f"The discussion contains {len(G.nodes())} argument elements with {len(G.edges())} logical connections.")
    
    # Most common argument types
    if arg_counts:
        most_common = arg_counts.most_common(1)[0]
        insights.append(f"The most common argument type is '{most_common[0]}' ({most_common[1]} instances).")
    
    # Analyze claim-evidence ratio
    claims = arg_counts.get('claim', 0) + arg_counts.get('counterclaim', 0)
    evidence = arg_counts.get('evidence', 0)
    
    if claims > 0:
        evidence_ratio = evidence / claims
        if evidence_ratio < 0.5:
            insights.append(f"The discussion has a low evidence-to-claim ratio ({evidence_ratio:.2f}), suggesting many claims lack supporting evidence.")
        elif evidence_ratio > 1.5:
            insights.append(f"The discussion has a high evidence-to-claim ratio ({evidence_ratio:.2f}), indicating claims are well-supported.")
    
    # Controversial claims
    if controversial:
        insights.append(f"Found {len(controversial)} controversial claims that received both agreement and disagreement.")
    
    # Generate summary text
    summary = f"""
## Argumentation Analysis

### Overview
This visualization maps the logical structure of the discussion, showing how arguments relate to each other.
Hover over a node to see detailed information.

### Key Statistics
- **Total Arguments**: {len(G.nodes())}
- **Logical Connections**: {len(G.edges())}
- **Argument Types**: {', '.join(f"{k} ({v})" for k, v in arg_counts.most_common())}

### Key Insights
{chr(10).join(f"- {insight}" for insight in insights)}

### Interpretation Guide
- **Blue nodes**: Primary claims or assertions
- **Red nodes**: Counterclaims that challenge other arguments
- **Green nodes**: Evidence or supporting information
- **Purple nodes**: Questions that probe or challenge
- **Cyan nodes**: Agreements with previous statements
- **Pink nodes**: Disagreements with previous points
- **Orange nodes**: Clarifications or elaborations

Node size reflects message length and importance, while connections show how arguments respond to each other.
"""
    
    return fig, analysis, summary
