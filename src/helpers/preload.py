
import streamlit as st
import time

def eager_load_heavy_libs():
    """
    Eagerly import heavy libraries to avoid lag when switching tabs
    --> moves the loading time to the inital app startup
    """
    
    with st.spinner("Loading AI libraries (Torch, Cellpose, SAM2)..."):
        import torch
        import torch.nn
        import torch.optim
        import torch.utils.data
        import torchvision
        import torchvision.models
        import torchvision.transforms
        
        import cv2
        import numpy as np
        import pandas as pd
        import scipy.ndimage
        from PIL import Image, ImageDraw
        import io
        import os
        import tempfile
        import zipfile
        
        import sklearn.model_selection
        import sklearn.utils.class_weight
        import sklearn.metrics

        # IO and Components
        import tifffile
        import huggingface_hub
        try:
            import streamlit_image_coordinates
        except ImportError:
            pass

        import cellpose.models
        import cellpose.core
        import cellpose.io
        import cellpose.train
        import cellpose.metrics
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            pass 
            
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.io
        