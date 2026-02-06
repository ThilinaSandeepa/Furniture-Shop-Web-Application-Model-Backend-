import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import logging
from sqlalchemy import create_engine


class RoomPlannerModel:
    def __init__(self, db_config):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger('RoomPlannerModel')

        db_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        self.engine = create_engine(db_url, pool_pre_ping=True)

        # === NEW: Feature name mapping for flexibility ===
        self.FEATURE_NAME_MAPPING = {
            'room_size': ['room_size', 'room size', 'roomsize', 'Room Size', 'RoomSize', 'ROOM_SIZE'],
            'style': ['style', 'Style', 'design_style', 'Design Style', 'design style', 'DESIGN_STYLE', 'STYLE']
            # You can add more later if needed
        }

        # Reverse mapping: normalized_name -> canonical_name
        self.reverse_mapping = {}
        for canonical, variations in self.FEATURE_NAME_MAPPING.items():
            for var in variations:
                self.reverse_mapping[var.lower().replace(' ', '_')] = canonical

        self.df = self._load_dataset()
        self.encoder = None
        self.feature_matrix = None
        self._prepare_similarity_features()

    def _load_dataset(self):
        try:
            with self.engine.connect() as connection:
                df_products = pd.read_sql("""
                    SELECT id, name, description, price, category_id
                    FROM products 
                    WHERE is_deleted = 0
                """, connection)

                df_categories = pd.read_sql("""
                    SELECT id, name AS room_type
                    FROM categories 
                    WHERE is_deleted = 0
                """, connection)

                df_features = pd.read_sql("""
                    SELECT product_id, feature_name, feature_value
                    FROM product_features 
                    WHERE is_deleted = 0
                """, connection)

                df_images = pd.read_sql("""
                    SELECT product_id, image_path
                    FROM product_images 
                    WHERE is_deleted = 0
                    ORDER BY createdAt ASC
                """, connection)

            df = df_products.merge(df_categories, left_on='category_id', right_on='id', how='left')
            df = df.rename(columns={'id_x': 'id'}).drop(columns=['id_y', 'category_id'], errors='ignore')

            # === IMPROVED FEATURE PIVOTING WITH NAME NORMALIZATION ===
            if not df_features.empty:
                df_features = df_features.drop_duplicates(subset=['product_id', 'feature_name'], keep='first')

                # Normalize feature_name before pivoting
                df_features['normalized_name'] = df_features['feature_name'].str.lower().str.replace(' ', '_')

                # Map to canonical names using reverse mapping
                df_features['canonical_name'] = df_features['normalized_name'].map(self.reverse_mapping)

                # Keep only recognized features, drop others silently
                df_features = df_features.dropna(subset=['canonical_name'])

                # Now pivot using canonical_name
                pivoted = df_features.pivot(index='product_id', columns='canonical_name', values='feature_value').reset_index()

                # Merge into main df
                df = df.merge(pivoted, left_on='id', right_on='product_id', how='left')
                df = df.drop(columns=['product_id'], errors='ignore')

            # === IMAGE HANDLING (unchanged) ===
            if not df_images.empty:
                df_images = df_images.drop_duplicates(subset=['product_id'], keep='first')
                df = df.merge(df_images, left_on='id', right_on='product_id', how='left')
                df['image'] = df['image_path']
                df = df.drop(columns=['image_path', 'product_id'], errors='ignore')
            else:
                df['image'] = 'default.jpg'

            df['id'] = df['id'].astype(str)

            # === BUDGET RANGE (you already updated this) ===
            if 'budget_range' not in df.columns:
                df['budget_range'] = pd.cut(
                    df['price'],
                    bins=[0, 80000, 200000, float('inf')],
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True
                ).astype(str)

            # === FALLBACK FOR MISSING COLUMNS (now more robust) ===
            for col in ['room_size', 'style', 'room_type']:
                if col not in df.columns:
                    df[col] = 'Unknown'
                df[col] = df[col].fillna('Unknown').str.capitalize()

            df = df.drop_duplicates(subset=['id']).reset_index(drop=True)

            self.logger.info(f"Successfully loaded {len(df)} products")
            return df

        except Exception as e:
            self.logger.error(f"Database load error: {str(e)}")
            raise Exception(f"Failed to load data: {str(e)}")

    def _prepare_similarity_features(self):
        categorical_cols = ['room_size', 'room_type', 'style', 'budget_range']

        for col in categorical_cols:
            if col not in self.df.columns:
                self.df[col] = 'Unknown'

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = self.encoder.fit_transform(self.df[categorical_cols])

        price_norm = (self.df['price'] / self.df['price'].max()).values.reshape(-1, 1)
        price_weighted = np.repeat(price_norm, 3, axis=1)

        self.feature_matrix = np.hstack((encoded, price_weighted))

    def search(self, room_size, room_type, style, budget_range, limit=5):
        self.df = self._load_dataset()
        self._prepare_similarity_features()

        room_size = room_size.capitalize()
        room_type = room_type.replace("-", " ").title()
        style = style.capitalize()
        budget_range = budget_range.capitalize()

        exact_df = self.df[
            (self.df['room_size'] == room_size) &
            (self.df['room_type'] == room_type) &
            (self.df['style'] == style) &
            (self.df['budget_range'] == budget_range)
        ].copy()

        result_df = exact_df

        if len(result_df) < limit:
            query_vector = self._get_query_vector(room_size, room_type, style, budget_range)
            similarities = cosine_similarity(query_vector, self.feature_matrix)[0]

            candidates = self.df[self.df['room_type'] == room_type].copy()
            candidate_indices = candidates.index
            candidates['match_score'] = [similarities[i] * 100 for i in candidate_indices]
            candidates = candidates[~candidates.index.isin(exact_df.index)]

            similar_df = candidates.sort_values('match_score', ascending=False).head(limit * 2)
            result_df = pd.concat([exact_df, similar_df]).drop_duplicates(subset=['id'])

        result_df = result_df[result_df['room_type'] == room_type]

        if len(result_df) == 0:
            result_df = self.df[self.df['room_type'] == room_type].head(limit)

        if 'match_score' not in result_df.columns:
            result_df['match_score'] = 80.0
        exact_mask = (
            (result_df['room_size'] == room_size) &
            (result_df['room_type'] == room_type) &
            (result_df['style'] == style) &
            (result_df['budget_range'] == budget_range)
        )
        result_df.loc[exact_mask, 'match_score'] = 100.0

        result_df = result_df.sort_values('match_score', ascending=False).head(limit)

        suggestions = [
            {
                'id': row['id'],
                'name': row['name'],
                'price': float(row['price']),
                'description': row['description'],
                'image': row['image'],
                'link': f"/product/{row['id']}",
                'match_score': round(row['match_score'], 2)
            }
            for _, row in result_df.iterrows()
        ]

        return {
            'title': f"{style} {room_type} Design",
            'description': f"Recommendations for your {room_size.lower()} {room_type.lower()} in {style.lower()} style ({budget_range.lower()} budget):",
            'suggestions': suggestions,
            'tips': self._generate_design_tips(room_size, room_type, style, budget_range),
            'tags': self._generate_tags(room_size, room_type, style, budget_range)
        }

    def _get_query_vector(self, room_size, room_type, style, budget_range):
        query_df = pd.DataFrame([{
            'room_size': room_size,
            'room_type': room_type,
            'style': style,
            'budget_range': budget_range
        }])

        encoded_query = self.encoder.transform(query_df)

        mask = (self.df['room_type'] == room_type) & (self.df['budget_range'] == budget_range)
        avg_price = self.df[mask]['price'].mean()
        if np.isnan(avg_price):
            avg_price = self.df['price'].mean()

        price_norm = np.array([[avg_price / self.df['price'].max()]])
        price_weighted = np.repeat(price_norm, 3, axis=1)

        return np.hstack((encoded_query, price_weighted))

    def _generate_design_tips(self, room_size, room_type, style, budget_range):
        tips = []
        if room_size == "Small":
            tips += ["Use multi-functional furniture", "Opt for light colors", "Use mirrors to enhance space"]
        elif room_size == "Medium":
            tips += ["Balance furniture scale", "Define zones with rugs", "Create a focal point"]
        else:
            tips += ["Use large statement pieces", "Create conversation areas", "Layer lighting"]

        if style == "Modern":
            tips += ["Clean lines", "Metal & glass accents", "Neutral with bold pops"]
        elif style == "Classic":
            tips += ["Rich woods", "Symmetrical layout", "Elegant details"]
        else:
            tips += ["Less is more", "Hidden storage", "Neutral palette"]

        if budget_range == "Low":
            tips += ["Prioritize essentials", "Mix vintage finds"]
        elif budget_range == "Medium":
            tips += ["Invest in key pieces", "Mix high/low"]
        else:
            tips += ["Choose timeless quality", "Custom options available"]

        return tips

    def _generate_tags(self, room_size, room_type, style, budget_range):
        base = [room_size, room_type, style, f"{budget_range} Budget"]
        extras = {"Modern": ["Contemporary", "Sleek"], "Classic": ["Elegant", "Timeless"], "Minimalist": ["Clean", "Simple"]}.get(style, [])[:2]
        room_extras = {"Living Room": ["Cozy", "Entertainment"], "Bedroom": ["Restful", "Relaxation"], "Dining": ["Gathering", "Meals"]}.get(room_type, [])[:2]
        return base + extras + room_extras

    def get_product_by_id(self, product_id):
        self.df = self._load_dataset()
        self._prepare_similarity_features()

        product_id = str(product_id)
        product_row = self.df[self.df['id'] == product_id]

        if product_row.empty:
            return None

        product = product_row.iloc[0].to_dict()
        return {'product': product}

    def get_related_products(self, product_id, limit=10):
        """
        ML-powered related products using cosine similarity on the same feature space.
        """
        self.df = self._load_dataset()
        self._prepare_similarity_features()

        product_id = str(product_id)
        product_row = self.df[self.df['id'] == product_id]

        if product_row.empty:
            return []

        target_idx = product_row.index[0]
        target_vector = self.feature_matrix[target_idx].reshape(1, -1)

        # Compute similarity to all other products
        similarities = cosine_similarity(target_vector, self.feature_matrix)[0]
        similarities[target_idx] = -1  # Exclude itself

        # Get top similar indices
        top_indices = np.argsort(similarities)[::-1][:limit]
        related_df = self.df.iloc[top_indices]

        related_products = [
            {
                'id': row['id'],
                'name': row['name'],
                'price': float(row['price']),
                'description': row['description'],
                'image': row['image'],
                'link': f"/product/{row['id']}",
                'match_score': round(similarities[idx] * 100, 2)
            }
            for idx, (_, row) in zip(top_indices, related_df.iterrows())
        ]

        return related_products

    def get_available_filters(self):
        self.df = self._load_dataset()
        return {
            'room_sizes': sorted(self.df['room_size'].unique().tolist()) if 'room_size' in self.df else [],
            'room_types': sorted(self.df['room_type'].unique().tolist()),
            'styles': sorted(self.df['style'].unique().tolist()) if 'style' in self.df else [],
            'budget_ranges': sorted(self.df['budget_range'].unique().tolist()),
            'price_range': {
                'min': float(self.df['price'].min()),
                'max': float(self.df['price'].max())
            }
        }