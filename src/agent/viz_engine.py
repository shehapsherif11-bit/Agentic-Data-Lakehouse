"""
Visualization engine using Plotly for the analytical AI pipeline.
"""
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import logging

logger = logging.getLogger(__name__)

def safe_chart_data(data: list[dict], x_col: str, y_col: str, color_col: str = None) -> tuple[list, list, list]:
    """Extract and clean data from query results. Handle missing columns and casing gracefully."""
    if not data or not isinstance(data, list):
        return [], [], []
        
    # Get actual column names from the first row to handle case insensitivity
    first_row = data[0] if isinstance(data[0], dict) else {}
    col_map = {k.lower(): k for k in first_row.keys()}
    
    actual_x = col_map.get(x_col.lower()) if x_col else None
    actual_y = col_map.get(y_col.lower()) if y_col else None
    actual_color = col_map.get(color_col.lower()) if color_col else None
    
    if not actual_x or not actual_y:
        return [], [], []
        
    x_data = []
    y_data = []
    color_data = []
    
    for row in data:
        if isinstance(row, dict) and actual_x in row and actual_y in row:
            x_data.append(row[actual_x])
            y_data.append(row[actual_y])
            if actual_color and actual_color in row:
                color_data.append(row[actual_color])
            
    return x_data, y_data, color_data

def generate_chart(viz_spec: dict, data: list[dict]) -> str | None:
    """Generate a Plotly chart from a structured viz spec and query result data."""
    if not viz_spec.get('should_visualize', False):
        return None
        
    if not data or not isinstance(data, list) or len(data) == 0:
        logger.warning("No data provided for visualization.")
        return None
        
    chart_type = viz_spec.get('chart_type')
    x_axis = viz_spec.get('x_axis')
    y_axis = viz_spec.get('y_axis')
    color_col = viz_spec.get('color_col')
    title = viz_spec.get('title', 'Data Visualization')
    
    if not chart_type or not x_axis or not y_axis:
        logger.warning("Incomplete viz_spec provided.")
        return None
        
    try:
        x_data, y_data, color_data = safe_chart_data(data, x_axis, y_axis, color_col)
        
        if not x_data or not y_data:
            logger.warning("Could not extract x or y data using specified columns.")
            return None
            
        fig = None
        template = "plotly_white"
        
        # Prepare kwargs for px functions
        kwargs = {
            'x': x_data,
            'y': y_data,
            'title': title,
            'labels': {'x': x_axis, 'y': y_axis},
            'template': template
        }
        if color_data:
            kwargs['color'] = color_data
            kwargs['labels']['color'] = color_col
        
        if chart_type == 'line':
            fig = px.line(**kwargs)
        elif chart_type == 'bar':
            fig = px.bar(**kwargs)
        elif chart_type == 'stacked_bar':
            fig = px.bar(barmode='stack', **kwargs)
        elif chart_type == 'scatter':
            fig = px.scatter(**kwargs)
        elif chart_type == 'pie':
            fig = px.pie(names=x_data, values=y_data, title=title, template=template)
        else:
            logger.warning(f"Unsupported chart_type: {chart_type}")
            return None
            
        if fig:
            return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        return None
        
    return None
