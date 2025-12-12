"""Streamlit dashboard for Reddit TV show discussion analytics."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


@st.cache_data
def load_data():
    """Load all CSV data files.

    Returns:
        Tuple of dataframes: (comments, entities, theories, scenes,
                             comment_entities, comment_theories, sentiments)
    """
    data_dir = Path("data/csv_exports")

    comments = pd.read_csv(data_dir / "comments.csv")
    entities = pd.read_csv(data_dir / "entities.csv")
    theories = pd.read_csv(data_dir / "theories.csv")
    scenes = pd.read_csv(data_dir / "scenes.csv")
    comment_entities = pd.read_csv(data_dir / "comment_entities.csv")
    comment_theories = pd.read_csv(data_dir / "comment_theories.csv")
    sentiments = pd.read_csv(data_dir / "comment_sentiments.csv")

    return comments, entities, theories, scenes, comment_entities, comment_theories, sentiments


def create_character_popularity_chart(comment_entities: pd.DataFrame, entities: pd.DataFrame) -> go.Figure:
    """Create line chart showing character popularity over episodes.

    Args:
        comment_entities: Comment-entity link dataframe
        entities: Entity reference dataframe

    Returns:
        Plotly figure object
    """
    # Filter to characters only
    character_names = entities[entities['type'] == 'character']['canonical_name'].tolist()
    char_mentions = comment_entities[comment_entities['entity_canonical_name'].isin(character_names)]

    # Count mentions per episode per character
    popularity = char_mentions.groupby(['episode', 'entity_canonical_name']).size().reset_index(name='mentions')

    # Get top 5 characters by total mentions
    top_chars = popularity.groupby('entity_canonical_name')['mentions'].sum().nlargest(5).index
    popularity_top = popularity[popularity['entity_canonical_name'].isin(top_chars)]

    # Create line chart
    fig = px.line(
        popularity_top,
        x='episode',
        y='mentions',
        color='entity_canonical_name',
        title='Character Popularity Over Episodes',
        labels={'episode': 'Episode', 'mentions': 'Mention Count', 'entity_canonical_name': 'Character'},
        markers=True
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=2, dtick=1),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_theory_support_chart(theories: pd.DataFrame) -> go.Figure:
    """Create bar chart showing theory support distribution by category.

    Args:
        theories: Theory reference dataframe

    Returns:
        Plotly figure object
    """
    # Count theories by category and consensus
    theory_dist = theories.groupby(['category', 'consensus']).size().reset_index(name='count')

    fig = px.bar(
        theory_dist,
        x='category',
        y='count',
        color='consensus',
        title='Theory Distribution by Category and Support Level',
        labels={'category': 'Theory Category', 'count': 'Number of Theories', 'consensus': 'Consensus'},
        barmode='group',
        color_discrete_map={'strong_support': '#2ca02c', 'moderate_support': '#ff7f0e', 'weak_support': '#d62728'}
    )

    fig.update_xaxes(tickangle=45)

    return fig


def create_sentiment_pie_chart(sentiments: pd.DataFrame) -> go.Figure:
    """Create pie chart showing sentiment distribution.

    Args:
        sentiments: Sentiment dataframe

    Returns:
        Plotly figure object
    """
    # Count sentiments
    sentiment_counts = sentiments['sentiment'].value_counts().head(10).reset_index()
    sentiment_counts.columns = ['sentiment', 'count']

    fig = px.pie(
        sentiment_counts,
        values='count',
        names='sentiment',
        title='Top 10 Sentiments in Comments',
        hole=0.3
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig


def create_entity_heatmap(comment_entities: pd.DataFrame, entities: pd.DataFrame) -> go.Figure:
    """Create heatmap showing entity mentions across episodes.

    Args:
        comment_entities: Comment-entity link dataframe
        entities: Entity reference dataframe

    Returns:
        Plotly figure object
    """
    # Filter to characters only
    character_names = entities[entities['type'] == 'character']['canonical_name'].tolist()
    char_mentions = comment_entities[comment_entities['entity_canonical_name'].isin(character_names)]

    # Create pivot table
    heatmap_data = char_mentions.groupby(['entity_canonical_name', 'episode']).size().reset_index(name='mentions')
    pivot = heatmap_data.pivot(index='entity_canonical_name', columns='episode', values='mentions').fillna(0)

    # Get top 10 entities
    pivot = pivot.loc[pivot.sum(axis=1).nlargest(10).index]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Ep {i}" for i in pivot.columns],
        y=pivot.index,
        colorscale='Blues',
        hovertemplate='<b>%{y}</b><br>%{x}: %{z} mentions<extra></extra>'
    ))

    fig.update_layout(
        title='Entity Mention Heatmap (Top 10 Characters)',
        xaxis_title='Episode',
        yaxis_title='Character',
        height=400
    )

    return fig


def search_comments(
    comments: pd.DataFrame,
    comment_entities: pd.DataFrame,
    comment_theories: pd.DataFrame,
    search_text: str = "",
    episode_filter: str = "All",
    entity_filter: str = "All",
    theory_filter: str = "All"
) -> pd.DataFrame:
    """Search and filter comments based on criteria.

    Args:
        comments: Comments dataframe
        comment_entities: Comment-entity links
        comment_theories: Comment-theory links
        search_text: Text to search in comment body
        episode_filter: Episode number or "All"
        entity_filter: Entity name or "All"
        theory_filter: Theory ID or "All"

    Returns:
        Filtered comments dataframe
    """
    result = comments.copy()

    # Text search
    if search_text:
        result = result[result['body'].str.contains(search_text, case=False, na=False)]

    # Episode filter
    if episode_filter != "All":
        result = result[result['episode'] == int(episode_filter)]

    # Entity filter
    if entity_filter != "All":
        comment_ids = comment_entities[comment_entities['entity_canonical_name'] == entity_filter]['comment_id'].unique()
        result = result[result['comment_id'].isin(comment_ids)]

    # Theory filter
    if theory_filter != "All":
        comment_ids = comment_theories[comment_theories['theory_id'] == int(theory_filter)]['comment_id'].unique()
        result = result[result['comment_id'].isin(comment_ids)]

    return result


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Severance Reddit Analytics", layout="wide")

    st.title("📊 Severance Reddit Discussion Analytics")
    st.markdown("Season 1 Episodes 2-9 Analysis")

    # Load data
    with st.spinner("Loading data..."):
        comments, entities, theories, scenes, comment_entities, comment_theories, sentiments = load_data()

    # Sidebar
    st.sidebar.header("📈 Dataset Summary")
    st.sidebar.metric("Total Comments", f"{len(comments):,}")
    st.sidebar.metric("Entities", len(entities))
    st.sidebar.metric("Theories", len(theories))
    st.sidebar.metric("Scenes", len(scenes))
    st.sidebar.metric("Entity Mentions", f"{len(comment_entities):,}")
    st.sidebar.metric("Theory Mentions", f"{len(comment_theories):,}")

    # Main visualizations
    st.header("📊 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Creating character popularity chart..."):
            fig1 = create_character_popularity_chart(comment_entities, entities)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        with st.spinner("Creating theory support chart..."):
            fig2 = create_theory_support_chart(theories)
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        with st.spinner("Creating sentiment distribution..."):
            fig3 = create_sentiment_pie_chart(sentiments)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        with st.spinner("Creating entity heatmap..."):
            fig4 = create_entity_heatmap(comment_entities, entities)
            st.plotly_chart(fig4, use_container_width=True)

    # Comment search section
    st.header("🔍 Search Comments")

    search_col1, search_col2, search_col3, search_col4 = st.columns([3, 1, 2, 2])

    with search_col1:
        search_text = st.text_input("Search text", placeholder="Enter keywords...")

    with search_col2:
        episode_options = ["All"] + [str(i) for i in range(2, 10)]
        episode_filter = st.selectbox("Episode", episode_options)

    with search_col3:
        entity_options = ["All"] + sorted(entities['canonical_name'].tolist())
        entity_filter = st.selectbox("Entity", entity_options)

    with search_col4:
        theory_options = ["All"] + [f"{t['theory_id']}: {t['canonical_claim'][:30]}..." for _, t in theories.iterrows()]
        theory_filter_display = st.selectbox("Theory", theory_options)
        theory_filter = "All" if theory_filter_display == "All" else theory_filter_display.split(":")[0]

    # Perform search
    if st.button("Search") or search_text or episode_filter != "All" or entity_filter != "All" or theory_filter != "All":
        with st.spinner("Searching comments..."):
            results = search_comments(
                comments,
                comment_entities,
                comment_theories,
                search_text,
                episode_filter,
                entity_filter,
                theory_filter
            )

        st.subheader(f"Results: {len(results)} comments")

        if len(results) > 0:
            # Add entities mentioned to results
            results_with_entities = results.copy()

            # Get entities for each comment
            def get_entities(comment_id):
                ents = comment_entities[comment_entities['comment_id'] == comment_id]['entity_canonical_name'].tolist()
                return ', '.join(ents[:3]) + ('...' if len(ents) > 3 else '')

            results_with_entities['entities_mentioned'] = results_with_entities['comment_id'].apply(get_entities)

            # Pagination
            page_size = 20
            total_pages = (len(results_with_entities) - 1) // page_size + 1

            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            else:
                page = 1

            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size

            page_results = results_with_entities.iloc[start_idx:end_idx]

            # Display results
            for idx, row in page_results.iterrows():
                with st.expander(f"**Episode {row['episode']}** | u/{row['author']} | Score: {row['score']}"):
                    st.write(row['body'])
                    if row['entities_mentioned']:
                        st.caption(f"**Entities mentioned:** {row['entities_mentioned']}")
        else:
            st.info("No comments found matching your search criteria.")


if __name__ == "__main__":
    main()
