import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import ast

from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import numpy as np

# Calculate Haversine distance between two latitude/longitude points
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Convert haversine distance threshold to radians
def collapse_nodes(geo_df, threshold_km=50):
    geo_df['lat'] = pd.to_numeric(geo_df['lat'], errors='coerce')
    geo_df['lon'] = pd.to_numeric(geo_df['lon'], errors='coerce')
    geo_df = geo_df.dropna(subset=['lat', 'lon'])

    coords = geo_df[['lat', 'lon']].values
    db = DBSCAN(eps=threshold_km / 6371, min_samples=1, metric='haversine').fit(np.radians(coords))
    geo_df['cluster'] = db.labels_

    # Collapse: take first node per cluster for now (can also use mean or central node)
    collapsed_df = geo_df.groupby('cluster').agg({
        'lat': 'mean',
        'lon': 'mean',
        'place': lambda x: ', '.join(sorted(set(x))),
        'asn': lambda x: next(iter(x.dropna()), '*')  # First non-null ASN
    }).reset_index(drop=True)

    return collapsed_df

# The actual plotting function
def plot_geographical_path(forward_path, src_lat, src_lon, src_city, asn_color_map, is_reverse=False):
    geo_hops = [(hop.get('place', 'Missing'), hop.get('latitude'), hop.get('longitude'), hop.get('associated_asn'))
                for hop in forward_path if hop.get('latitude') and hop.get('longitude')]

    if not geo_hops:
        st.warning("No valid lat/lon data found in the forward path.")
        return

    geo_df = pd.DataFrame(geo_hops, columns=['place', 'lat', 'lon', 'asn'])
    collapsed_df = collapse_nodes(geo_df, threshold_km=50)
    print(geo_df)
    fig_map = go.Figure()

    for i in range(len(collapsed_df) - 1):
        if i == 0:
            if is_reverse:
                text = [f"Client: {collapsed_df.place.iloc[i]}", collapsed_df.place.iloc[i+1]]
            if not is_reverse:
                text = [f"Server: {collapsed_df.place.iloc[i]}", collapsed_df.place.iloc[i + 1]]
        else:
            text = [collapsed_df.place.iloc[i], collapsed_df.place.iloc[i + 1]]

        asn = collapsed_df.asn.iloc[i+1]
        color = asn_color_map.get(asn, 'gray')

        fig_map.add_trace(go.Scattergeo(
            lon=[collapsed_df.lon.iloc[i], collapsed_df.lon.iloc[i + 1]],
            lat=[collapsed_df.lat.iloc[i], collapsed_df.lat.iloc[i + 1]],
            mode='lines+markers+text',
            line=dict(width=2, color=color),
            marker=dict(size=8, color=color),
            text=text,
            textposition='top center',
            textfont=dict(color='black'),
            showlegend=False
        ))


    if not is_reverse:
        last_hop = (collapsed_df.lat.iloc[-1], collapsed_df.lon.iloc[-1])
        client_loc = (src_lat, src_lon)
        distance_to_client = haversine(*last_hop, *client_loc)

        if distance_to_client > 50:  # adjustable threshold in km
            fig_map.add_trace(go.Scattergeo(
                lon=[collapsed_df.lon.iloc[-1], src_lon],
                lat=[collapsed_df.lat.iloc[-1], src_lat],
                mode='lines+markers+text',
                line=dict(width=2, color='red', dash='dash'),
                marker=dict(size=10, symbol='circle'),
                text=[collapsed_df.place.iloc[-1], f'Client: {src_city}'],
                textposition='top center',
                textfont=dict(color='black'),
                showlegend=False
            ))

    fig_map.update_layout(
        title="Geographical Path (Dashed Line to Client/Server)",
        geo=dict(
            scope="world",  # Show entire world
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showocean=True,
            oceancolor="rgb(210, 230, 250)",

            showcountries=True,  # <<< Show country borders
            countrycolor="rgb(204, 204, 204)",

            showsubunits=True,  # Subunits = states, provinces
            subunitcolor="rgb(220, 220, 220)",

            showlakes=True,  # <<< ADD lakes
            lakecolor="rgb(180, 210, 250)",

            showrivers=True,  # <<< (Optional) draw rivers too
            rivercolor="rgb(170, 200, 250)",

            showframe=False,  # <<< Remove the box around the map
            projection_type="natural earth"  # <<< (optional) a nice looking projection
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )

    st.plotly_chart(fig_map, use_container_width=True)

def build_asn_color_map(asn_values):
    unique_asns = list(set(asn_values))
    color_list = px.colors.qualitative.Plotly
    asn_color_map = {'*': 'gray'}  # Ensure default color for unknown

    for i, asn in enumerate([asn for asn in unique_asns if asn != '*']):
        asn_color_map[asn] = color_list[i % len(color_list)]

    return asn_color_map
def draw_combined_logical_path(forward_path, src_asn, asn_color_map, is_reverse=False):
    ttl_range = []
    ip_labels = []
    as_labels = []
    org_labels = []
    city_labels = []
    rtt_values = []
    fiber_km = []
    asn_values = []
    rdns_tooltips = []
    src_asn_entry_index = None

    for i, hop in enumerate(forward_path):
        ttl = int(hop.get('ttl', i+1))
        ip = hop.get('addr') or '*'
        asn = str(hop.get('associated_asn') or '*')
        org = hop.get('associated_org') or '*'
        city = hop.get('place') or '*'
        rdns = hop.get('rdns_name') or '*'
        rtt = hop.get('rtts', None)
        fiber = hop.get('speed_of_internet_fiber', None)

        if src_asn_entry_index is None and asn == str(src_asn):
            src_asn_entry_index = ttl

        ttl_range.append(ttl)
        ip_labels.append(ip)
        as_labels.append(asn)
        org_labels.append(org)
        city_labels.append(city)
        rdns_tooltips.append(rdns)
        rtt_values.append(rtt)
        fiber_km.append(fiber)
        asn_values.append(asn)

    # Color mapping functions
    def map_colors(values):
        return [asn_color_map.get(val, 'gray') for val in values]

    fig = go.Figure()
    ip_to_asn = dict(zip(ip_labels, as_labels))

    def map_ip_as_colors(ip_list):
        return [asn_color_map.get(ip_to_asn.get(ip, '*'), 'gray') for ip in ip_list]

    # IP row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['IP'] * len(ip_labels),
        mode='markers + text',
        marker=dict(size=12, symbol='circle', color=map_ip_as_colors(ip_labels)),
        text=ip_labels,
        hovertext=rdns_tooltips,
        textposition='top center',
        hoverinfo='text',
        name='IP'
    ))

    # AS row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['AS'] * len(as_labels),
        mode='markers + text',
        marker=dict(size=12, symbol='square', color=map_colors(as_labels)),
        text=as_labels,
        hovertext=rdns_tooltips,
        textposition='top center',
        hoverinfo='text',
        name='AS'
    ))

    # rDNS row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['rDNS'] * len(rdns_tooltips),
        mode='markers',
        marker=dict(size=12, symbol='pentagon-open-dot', color='orange'),
        textposition='top center',
        hoverinfo='text',
        name='rDNS'
    ))

    # Org row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['Org'] * len(org_labels),
        mode='markers',
        marker=dict(size=12, symbol='diamond', color='purple'),
        text=org_labels,
        hovertext=rdns_tooltips,
        hoverinfo='text',
        textposition='top center',
        name='Org'
    ))

    # City row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['City'] * len(city_labels),
        mode='markers',
        marker=dict(size=12, symbol='hexagon', color='green'),
        text=city_labels,
        textposition='top center',
        hoverinfo='text',
        name='City'
    ))

    # RTT row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['RTT'] * len(rtt_values),
        mode='markers+text',
        marker=dict(size=12, symbol='circle', color=map_colors(as_labels)),
        text=[f"{r} ms" if r else "?" for r in rtt_values],
        hoverinfo='text',
        textposition='top center',
        name='RTT'
    ))

    # Fiber row
    fig.add_trace(go.Scatter(
        x=ttl_range,
        y=['Fiber Distance'] * len(fiber_km),
        mode='markers+text',
        marker=dict(size=12, symbol='circle-open', color=map_colors(as_labels)),
        text=[f"{float(d):.1f} ms" if d else "?" for d in fiber_km],
        hoverinfo='text',
        textposition='top center',
        name='Optimal Fiber Speed'
    ))

    # Vertical line for ASN entry
    if src_asn_entry_index:
        annotation_text = "Leaving src ASN" if is_reverse else "Entered src ASN"
        fig.add_vline(
            x=src_asn_entry_index,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=annotation_text,
            annotation_position="top left"
        )

    annotations = []

    for i in range(len(ttl_range)):
        # City-level annotation
        annotations.append(go.layout.Annotation(
            x=ttl_range[i],
            y='City',
            text=city_labels[i],
            textangle=30,
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=12)
        ))

        # Org-level annotation
        annotations.append(go.layout.Annotation(
            x=ttl_range[i],
            y='Org',
            text=org_labels[i],
            textangle=30,
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=12)
        ))

        # rDNS annotation
        annotations.append(go.layout.Annotation(
            x=ttl_range[i],
            y='rDNS',
            text=rdns_tooltips[i],
            textangle=30,
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=12)
        ))
    fig.update_layout(annotations=annotations)
    # Axis labels
    fig.update_layout(
        title="Logical Path: IP / rDNS / AS / Org / City / RTT / Fiber by TTL",
        xaxis_title="Hop TTL",
        yaxis=dict(
            title='',
            tickfont=dict(size=18),
            categoryorder='array',
            categoryarray=['Fiber Distance', 'RTT', 'City', 'Org', 'AS', 'rDNS', 'IP']
        ),
        xaxis=dict(tickfont=dict(size=18)),
        showlegend=False,
        height=750,
        width=1400,
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
def clean_and_parse_path(raw_value):
    try:
        # If it's already a list, we assume it's parsed JSON
        if isinstance(raw_value, list):
            return {'forward_updated_node_details': raw_value}

        # If it's a string, clean and parse it
        cleaned = raw_value.strip()

        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        cleaned = cleaned.replace("'", '"')

        return json.loads(cleaned)

    except Exception as e:
        st.error(f"Failed to parse path: {e}")
        return {'forward_updated_node_details': [], 'reverse_updated_node_details': []}

st.set_page_config(layout="wide")
st.title("HERMES Path Visualizer")

uploaded_file = st.file_uploader("Upload your HERMES file (CSV or JSONL)", type=["csv", "json"])


if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext == 'csv':
        df = pd.read_csv(uploaded_file)

    elif file_ext == 'json':
        # Read JSONL as lines and parse each JSON object
        try:
            raw_lines = uploaded_file.read().decode("utf-8").splitlines()
            json_objects = [json.loads(line) for line in raw_lines]
            df = pd.json_normalize(json_objects)
        except Exception as e:
            st.error(f"Failed to parse JSONL: {e}")
            st.stop()

    else:
        st.error("Unsupported file type.")
        st.stop()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())


    row_idx = st.number_input("Select row index to visualize", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[row_idx]
    partition_date = pd.to_datetime(row['partition_date']).date()
    st.markdown("## 🔗 Measurement Flow: Source → Destination")

    # Combine source ASN and city into one label
    df['source'] = df['src_asn'].astype(str) + ' - ' + df['src_city'].astype(str)

    # Determine whether the group has a high anomaly ratio
    anomaly_status = (
        df.groupby(['source', 'dst_site'])
        .agg({
            'anomaly_ratio_rtt': 'max',
            'anomaly_ratio_throughput': 'max'
        })
        .reset_index()
    )
    anomaly_status['has_anomaly'] = (
            (anomaly_status['anomaly_ratio_rtt'] > 0.8) |
            (anomaly_status['anomaly_ratio_throughput'] > 0.8)
    )

    # Merge with measurement counts
    sankey_data = (
        df.groupby(['source', 'dst_site'])
        .size()
        .reset_index(name='count')
        .merge(anomaly_status[['source', 'dst_site', 'has_anomaly']], on=['source', 'dst_site'])
    )

    # Create list of unique labels
    labels = list(pd.unique(sankey_data['source'].tolist() + sankey_data['dst_site'].tolist()))
    label_map = {label: i for i, label in enumerate(labels)}

    # Map to indices
    source_indices = sankey_data['source'].map(label_map)
    target_indices = sankey_data['dst_site'].map(label_map)
    values = sankey_data['count']

    # Assign colors
    link_colors = sankey_data['has_anomaly'].map(lambda x: "red" if x else "lightgray")

    # Plot
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])
    fig_sankey.update_layout(title_text="Source ASN + City → Destination Site (Red = Anomaly)", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)
    # --- SOURCE-DEST FILTER UI ---------------------------------------------------
    st.markdown("## 🔍 Filter by Source / Destination")

    # Source dropdown is built globally (not filtered by dst_site)
    src_options = (
        df[['src_asn', 'src_city']]
        .dropna()
        .drop_duplicates()
        .astype(str)
    )
    src_options['label'] = src_options['src_asn'] + ' – ' + src_options['src_city']
    src_map = dict(zip(src_options['label'], zip(src_options['src_asn'], src_options['src_city'])))

    selected_src = st.selectbox("Select Source ASN / City", list(src_map.keys()), key="src_selector")
    src_asn_filter, src_city_filter = src_map[selected_src]

    # Destination dropdown: allow selection even if only one option
    dst_options = df['dst_site'].dropna().unique().tolist()
    selected_dst = dst_options[0] if len(dst_options) == 1 else st.selectbox("Select Destination Server", dst_options,
                                                                             key="dst_selector")

    # Filter based on selected src/dst
    filtered_df = df[
        (df['src_asn'].astype(str) == src_asn_filter) &
        (df['src_city'].astype(str) == src_city_filter) &
        (df['dst_site'] == selected_dst)
        ]

    st.write(f"Selected destination: `{selected_dst}`")
    st.write(f"Source ASN: `{src_asn_filter}`, Source City: `{src_city_filter}`")
    st.write(f"Filtered rows: {len(filtered_df)}")

    from streamlit_plotly_events import plotly_events

    # --- TIME SERIES VISUALIZATION ----------------------------------------------
    st.markdown("## 📊 RTT and Throughput Over Time")

    if not filtered_df.empty:
        if "selected_row_index" not in st.session_state:
            st.session_state.selected_row_index = None
        filtered_df['measurement_timestamp'] = pd.to_datetime(filtered_df['window_start'])
        filtered_df = filtered_df.copy()
        filtered_df['timestamp_jittered'] = filtered_df['measurement_timestamp'] + pd.to_timedelta(
            filtered_df.groupby('measurement_timestamp').cumcount() * 1, unit='s'
        )
        rtt_fig = px.scatter(
            filtered_df,
            x='timestamp_jittered',
            y='ndt_rtt',
            hover_data=['id', 'ndt_throughput'],
            labels={'ndt_rtt': 'RTT (ms)', 'timestamp_jittered': 'Time'}
        )
        rtt_fig.update_traces(marker=dict(color='blue', size=8))

        # Baseline line
        rtt_fig.add_hline(
            y=filtered_df['baseline_median_rtt'].iloc[0],
            line_dash='dash',
            line_color='gray',
            annotation_text='Baseline RTT',
            annotation_position='top right'
        )

        # Add highlight marker if one is selected
        if st.session_state.selected_row_index is not None:
            highlight_time = filtered_df.iloc[st.session_state.selected_row_index]['measurement_timestamp']
            highlight_rtt = filtered_df.iloc[st.session_state.selected_row_index]['ndt_rtt']

            rtt_fig.add_trace(go.Scatter(
                x=[highlight_time],
                y=[highlight_rtt],
                mode='markers+text',
                marker=dict(color='red', size=12, symbol='star'),
                text=["Selected"],
                textposition="top center",
                showlegend=False
            ))
        rtt_fig.add_vrect(
            x0=partition_date,
            x1=partition_date + pd.Timedelta(days=1),
            fillcolor="orange",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Period of Interest",
            annotation_position="top left"
        )
        rtt_click = plotly_events(rtt_fig, click_event=True, hover_event=False, override_height=400)

        # --- Throughput Plot (Clickable)
        tp_fig = px.scatter(
            filtered_df,
            x='timestamp_jittered',
            y='ndt_throughput',
            hover_data=['id', 'ndt_rtt'],
            labels={'ndt_throughput': 'Throughput (Mbps)', 'timestamp_jittered': 'Time'}
        )
        tp_fig.update_traces(marker=dict(color='green', size=8))

        tp_fig.add_hline(
            y=filtered_df['baseline_median_throughput'].iloc[0],
            line_dash='dash',
            line_color='gray',
            annotation_text='Baseline Throughput',
            annotation_position='top right'
        )

        if st.session_state.selected_row_index is not None:
            highlight_time = filtered_df.iloc[st.session_state.selected_row_index]['timestamp_jittered']
            highlight_tp = filtered_df.iloc[st.session_state.selected_row_index]['ndt_throughput']

            tp_fig.add_trace(go.Scatter(
                x=[highlight_time],
                y=[highlight_tp],
                mode='markers+text',
                marker=dict(color='red', size=12, symbol='star'),
                text=["Selected"],
                textposition="top center",
                showlegend=False
            ))
        tp_fig.add_vrect(
            x0=partition_date,
            x1=partition_date + pd.Timedelta(days=1),
            fillcolor="orange",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Period of Interest",
            annotation_position="top left"
        )
        tp_click = plotly_events(tp_fig, click_event=True, hover_event=False, override_height=400)

        if rtt_click:
            point_index = rtt_click[0]['pointIndex']
            clicked_row = filtered_df.iloc[point_index]
            st.success(f"Selected row from RTT plot (index {point_index})")

        elif tp_click:
            point_index = tp_click[0]['pointIndex']
            clicked_row = filtered_df.iloc[point_index]
            st.success(f"Selected row from Throughput plot (index {point_index})")

        else:
            selected_id = st.selectbox("Fallback: Select Measurement ID", filtered_df['id'].tolist())
            clicked_row = filtered_df[filtered_df['id'] == selected_id].iloc[0]
        st.markdown("### 🎯 Selection Controls")

        if "selected_row_index" not in st.session_state:
            st.session_state.selected_row_index = None

        if st.button("🔄 Clear Selection"):
            st.session_state.selected_row_index = None
            st.rerun()
        # --- SHOW SELECTED ROW ---------------------------------------------------
        st.markdown("### 🔍 Selected Measurement Details")
        st.dataframe(clicked_row.to_frame().T)

        # --- SHOW METRICS -------------------------------------------------------
        st.markdown("---")
        st.markdown("## 📈 Performance Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("NDT RTT (in ms)", round(clicked_row['ndt_rtt'],2))
        with col2:
            st.metric("Throughput (in Mbps)", round(clicked_row['ndt_throughput'], 2))

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Measurements Baseline", int(clicked_row['measurement_count_per_site']))
        with col4:
            st.metric("Measurements On Day", int(clicked_row['current_number_of_measurements']))

        col5, col6 = st.columns(2)
        with col5:
            st.metric("Anomaly Ratio - RTT", f"{clicked_row['anomaly_ratio_rtt'] * 100:.1f}%")
        with col6:
            st.metric("Anomaly Ratio - Throughput", f"{clicked_row['anomaly_ratio_throughput'] * 100:.1f}%")

        col7, col8 = st.columns(2)
        with col7:
            st.metric("Baseline Median RTT (in ms)", round(clicked_row['baseline_median_rtt'],2))
        with col8:
            st.metric("Baseline Median Throughput (in Mbps)", round(clicked_row['baseline_median_throughput'], 2))

        col9, col10 = st.columns(2)
        with col9:
            st.metric("Reverse Geographic Distance (in km)", round(clicked_row['reverse_distance'], 2))
        with col10:
            st.metric("Forward Geographic Distance (in km)", round(clicked_row['forward_distance'], 2))

        # --- PARSE PATH AND VISUALIZE ------------------------------------------
        parsed_dict = clean_and_parse_path(clicked_row['forward_updated_node_details'])
        forward_path = parsed_dict.get('forward_updated_node_details', [])

        if forward_path:
            st.markdown("## 🔁 Forward Path Visualizations")
            asn_values = [str(hop.get('associated_asn') or '*') for hop in forward_path]
            asn_color_map = build_asn_color_map(asn_values)

            st.markdown("## 🔗 Logical Path (IP → rDNS → AS → Org → City)")
            draw_combined_logical_path(forward_path, clicked_row['src_asn'], asn_color_map)

            st.markdown("## 🌍 Geographical Path Map")
            plot_geographical_path(forward_path, clicked_row['src_lat'], clicked_row['src_lon'],
                                   clicked_row['src_city'], asn_color_map)
        else:
            st.warning("No valid forward path data found.")
        parsed_reverse_dict = clean_and_parse_path(clicked_row['reverse_updated_node_details'])
        reverse_path = parsed_reverse_dict.get('reverse_updated_node_details', [])
        if reverse_path:
            st.markdown("## 🔁 Reverse Path Visualizations")



            st.markdown("### 🔗 Logical Reverse Path (IP → rDNS → AS → Org → City)")
            draw_combined_logical_path(reverse_path, clicked_row['src_asn'], asn_color_map, is_reverse=True)

            st.markdown("### 🌍 Reverse Geographical Path Map")
            plot_geographical_path(reverse_path, clicked_row['dst_lat'], clicked_row['dst_lon'],
                                   clicked_row['dst_city'], asn_color_map, is_reverse = True)
        else:
            st.warning("No valid reverse path data found.")
    else:
        st.warning("No data for selected source/destination pair.")
