import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.services.video_processor import VideoProcessor
from src.utils.config import settings

st.set_page_config(
    page_title="Video Event Detection", 
    layout="wide",
    page_icon="🎥"
)

# Initialize video processor
@st.cache_resource
def get_video_processor():
    return VideoProcessor(lazy_load=False, heavy_models=True)

# Initialize with heavy model implementation - all models loaded
try:
    video_processor = get_video_processor()
    # Force check that all heavy models are loaded
    phase2_available = (hasattr(video_processor, 'phase2') and 
                       video_processor.phase2 is not None and 
                       video_processor.phase2_available)
    
    if not phase2_available:
        st.error("❌ HEAVY MODEL IMPLEMENTATION FAILED: BLIP-2 model not available")
        st.error("All AI models (OpenCLIP, BLIP-2, UniVTG) must be loaded for maximum accuracy")
        st.stop()
    else:
        st.success("🎉 HEAVY MODEL IMPLEMENTATION ACTIVE: All AI models loaded successfully!")
except Exception as e:
    st.error(f"Failed to initialize heavy model video processor: {e}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #3b82f6;
    margin-bottom: 1rem;
}
.result-card {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin-bottom: 1rem;
}
.confidence-high {
    color: #10b981;
    font-weight: bold;
}
.confidence-medium {
    color: #f59e0b;
    font-weight: bold;
}
.confidence-low {
    color: #ef4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🎥 Automatic Video Event Detection</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280;">Upload a video and describe the event you want to find using natural language!</p>', unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.selectbox(
    "Processing Mode",
    ["mvp", "reranked", "advanced"],
    index=1,
    help="MVP: Fast but basic, Reranked: Balanced accuracy/speed, Advanced: Most accurate"
)
top_k = st.sidebar.slider("Max Results", 1, 20, 10)
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.1)
debug_mode = st.sidebar.checkbox(
    "Debug Mode",
    value=False,
    help="Enable detailed logging and frame analysis"
)

# Mode descriptions - ALL HEAVY MODELS ACTIVE
mode_descriptions = {
    "mvp": "🚀 **Fast Mode**: Uses OpenCLIP for quick event detection",
    "reranked": "⚖️ **Heavy Mode**: OpenCLIP + BLIP-2 for maximum accuracy 🔥",
    "advanced": "🎯 **Ultra Mode**: Full pipeline with BLIP-2 + temporal refinement 🚀"
}
st.sidebar.markdown(mode_descriptions[mode])

# Show heavy model status
st.sidebar.success("🎉 **HEAVY MODEL IMPLEMENTATION ACTIVE**")
st.sidebar.info("✅ OpenCLIP: Loaded")
st.sidebar.info("✅ BLIP-2: Loaded")
st.sidebar.info("✅ All Models: Maximum Accuracy Mode")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">📁 Upload Video</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV (Max 2GB)"
    )
    
    if uploaded_file is not None:
        # Display video info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ Video uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Show video preview
        st.video(uploaded_file)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

with col2:
    st.markdown('<h2 class="sub-header">🔍 Event Query</h2>', unsafe_allow_html=True)
    query = st.text_area(
        "Describe the event you want to find:",
        placeholder="e.g., 'Two cars hit a man' or 'A person with a blue Honda wearing a dark green shirt'",
        height=100
    )
    
    # Example queries
    st.markdown("**💡 Example queries:**")
    example_queries = [
        "A person walking a dog",
        "Two cars colliding",
        "Someone wearing a red shirt",
        "A person falling down",
        "People shaking hands"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(f"📝 {example}", key=f"example_{i}"):
                query = example
                st.rerun()

# Process button
st.markdown("---")
process_col1, process_col2, process_col3 = st.columns([1, 2, 1])

with process_col2:
    process_button = st.button(
        "🚀 Find Events", 
        type="primary", 
        disabled=not (uploaded_file and query),
        use_container_width=True
    )

if process_button and uploaded_file and query:
    with st.spinner(f"🔍 Processing video with {mode} mode..."):
        try:
            # Validate video first
            validation_result = video_processor.validate_video(temp_video_path)
            
            if not validation_result['valid']:
                st.error(f"❌ {validation_result['error']}")
            else:
                # Process the query
                result = video_processor.process_query(
                    temp_video_path,
                    query,
                    mode=mode,
                    top_k=top_k,
                    threshold=threshold,
                    debug_mode=debug_mode
                )
                
                if result['status'] == 'error':
                    st.error(f"❌ Error: {result['error']}")
                else:
                    results = result['results']
                    
                    if not results:
                        st.warning("⚠️ No events found matching your query. Try lowering the confidence threshold or using a different query.")
                        
                        # Show debug suggestions if no events found
                        if debug_mode and 'debug_info' in result:
                            with st.expander("🔧 Debug Suggestions", expanded=True):
                                debug_info = result['debug_info']
                                similarities = [info['similarity'] for info in debug_info]
                                
                                if similarities:
                                    max_sim = max(similarities)
                                    mean_sim = sum(similarities) / len(similarities)
                                    
                                    st.write(f"**Current Analysis:**")
                                    st.write(f"- Maximum similarity score: {max_sim:.4f}")
                                    st.write(f"- Mean similarity score: {mean_sim:.4f}")
                                    st.write(f"- Current threshold: {threshold:.1f}")
                                    
                                    # Smart threshold recommendations
                                    st.write("\n**🎯 Smart Threshold Recommendations:**")
                                    
                                    # Recommend threshold slightly below max score
                                    if max_sim > 0.15:
                                        recommended_thresh = max_sim * 0.95
                                        count_at_recommended = sum(1 for s in similarities if s >= recommended_thresh)
                                        st.success(f"✅ **Recommended: {recommended_thresh:.3f}** (95% of max score) → {count_at_recommended} events")
                                    
                                    # Show percentile-based options
                                    percentiles = [90, 80, 70, 50]
                                    for p in percentiles:
                                        if len(similarities) > 0:
                                            thresh = sorted(similarities, reverse=True)[min(int(len(similarities) * (100-p) / 100), len(similarities)-1)]
                                            count = sum(1 for s in similarities if s >= thresh)
                                            if count > 0:
                                                st.write(f"  📊 {p}th percentile ({thresh:.4f}): {count} events")
                                    
                                    st.write("\n**💡 Try these actions:**")
                                    if max_sim < threshold:
                                        st.write(f"🔴 **Critical:** Your threshold ({threshold:.1f}) is higher than the maximum similarity ({max_sim:.4f})")
                                        st.write(f"   → Set threshold to {max_sim * 0.9:.3f} or lower")
                                    st.write("- Use more specific or general query terms")
                                    st.write("- Try different processing modes (MVP/Reranked/Advanced)")
                                    st.write("- Ensure the described event actually occurs in the video")
                                else:
                                    st.error("No similarity data available for analysis")
                    else:
                        st.success(f"✅ Found {len(results)} events!")
                        
                        # Display debug information if available
                        if debug_mode and 'debug_info' in result:
                            with st.expander("🔍 Debug Information", expanded=False):
                                debug_info = result['debug_info']
                                st.write(f"**Total windows processed:** {len(debug_info)}")
                                
                                # Show similarity statistics
                                similarities = [info['similarity'] for info in debug_info]
                                st.write(f"**Similarity range:** [{min(similarities):.6f}, {max(similarities):.6f}]")
                                st.write(f"**Mean similarity:** {sum(similarities)/len(similarities):.6f}")
                                
                                # Show top similarities
                                sorted_debug = sorted(debug_info, key=lambda x: x['similarity'], reverse=True)
                                st.write("**Top 5 similarity scores:**")
                                for i, info in enumerate(sorted_debug[:5]):
                                    st.write(f"  {i+1}. Window {info['window_index']}: {info['similarity']:.6f} at {info['timestamp']:.2f}s")
                        
                        # Display results
                        st.markdown('<h2 class="sub-header">📊 Results</h2>', unsafe_allow_html=True)
                        
                        for i, event_result in enumerate(results):
                            confidence = event_result['confidence']
                            
                            # Determine confidence color
                            if confidence >= 0.7:
                                conf_class = "confidence-high"
                                conf_emoji = "🟢"
                            elif confidence >= 0.5:
                                conf_class = "confidence-medium"
                                conf_emoji = "🟡"
                            else:
                                conf_class = "confidence-low"
                                conf_emoji = "🔴"
                            
                            with st.expander(f"Event {i+1} - {conf_emoji} Confidence: {confidence:.2f}", expanded=i==0):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    st.write(f"**⏰ Timestamp:** {event_result['timestamp']:.1f}s")
                                    st.write(f"**🔧 Processing Phase:** {event_result['phase']}")
                                    
                                    # Show caption if available (from phase 2)
                                    if 'caption' in event_result:
                                        st.write(f"**📝 Generated Caption:** {event_result['caption']}")
                                    
                                    # Show detailed scores if available
                                    if 'clip_score' in event_result:
                                        st.write(f"**🎯 CLIP Score:** {event_result['clip_score']:.3f}")
                                    if 'caption_score' in event_result:
                                        st.write(f"**📝 Caption Score:** {event_result['caption_score']:.3f}")
                                
                                with col2:
                                    st.metric("Confidence", f"{confidence:.2f}")
                                
                                with col3:
                                    # Download clip button
                                    if event_result.get('clip_path'):
                                        clip_path = Path(event_result['clip_path'])
                                        if clip_path.exists():
                                            with open(clip_path, 'rb') as clip_file:
                                                st.download_button(
                                                    f"📥 Download Clip {i+1}",
                                                    clip_file.read(),
                                                    file_name=f"event_{i+1}_{confidence:.2f}.mp4",
                                                    mime="video/mp4",
                                                    key=f"download_{i}"
                                                )
                                        else:
                                            st.warning("Clip file not found")
                                    else:
                                        st.info("Clip extraction failed")
                                
                                # Show video clip if available
                                if event_result.get('clip_path'):
                                    clip_path = Path(event_result['clip_path'])
                                    if clip_path.exists():
                                        # Read video file as bytes for Streamlit
                                        with open(clip_path, 'rb') as video_file:
                                            video_bytes = video_file.read()
                                        st.video(video_bytes)
                        
                        # Summary statistics
                        st.markdown("---")
                        st.markdown("### 📈 Summary")
                        
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        with summary_col1:
                            st.metric("Total Events", len(results))
                        
                        with summary_col2:
                            avg_confidence = sum(r['confidence'] for r in results) / len(results)
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                        with summary_col3:
                            max_confidence = max(r['confidence'] for r in results)
                            st.metric("Max Confidence", f"{max_confidence:.2f}")
                        
                        with summary_col4:
                            processing_mode = result['mode'].title()
                            st.metric("Processing Mode", processing_mode)
                
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
        finally:
            # Clean up temporary file
            if 'temp_video_path' in locals():
                try:
                    os.unlink(temp_video_path)
                except:
                    pass

elif not uploaded_file:
    st.info("👆 Please upload a video file to get started.")
elif not query:
    st.info("✍️ Please enter a query describing the event you want to find.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <h4>🤖 How it works</h4>
        <p>This system uses advanced AI models (OpenCLIP, BLIP-2) to automatically detect and extract specific events from videos based on your natural language description.</p>
        <p><strong>MVP Mode:</strong> Fast detection using OpenCLIP • <strong>Reranked Mode:</strong> Enhanced accuracy with BLIP-2 captioning • <strong>Advanced Mode:</strong> Full pipeline with temporal refinement</p>
    </div>
    """, 
    unsafe_allow_html=True
)