import pandas as pd
import numpy as np
import re
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
            'style': ['style', 'Style', 'design_style', 'Design Style', 'design style', 'DESIGN_STYLE', 'STYLE'],
            'room_type': ['room_type', 'room type', 'roomtype', 'Room Type', 'RoomType', 'ROOM_TYPE', 'room', 'Room']
        }

        # Category keyword → canonical room type mapping
        self.CATEGORY_ROOM_TYPE_MAP = [
            (['sofa', 'couch', 'loveseat', 'armchair', 'lounge', 'accent chair', 'living room', 'display cabinet', 'area rug', 'coffee table', 'tv unit', 'bookshelf', 'bookcase'], 'Living Room'),
            (['bed', 'bedroom', 'wardrobe', 'dresser', 'nightstand', 'mattress', 'dressing table', 'storage bench'], 'Bedroom'),
            (['dining', 'dining table', 'dining chair', 'dining set', 'buffet', 'sideboard'], 'Dining'),
        ]

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

            # Convert id to string early so merges work reliably across all id types
            df['id'] = df['id'].astype(str)

            # === IMPROVED FEATURE PIVOTING WITH NAME NORMALIZATION ===
            if not df_features.empty:
                df_features['product_id'] = df_features['product_id'].astype(str)
                df_features = df_features.drop_duplicates(subset=['product_id', 'feature_name'], keep='first')

                # Normalize feature_name before pivoting
                df_features['normalized_name'] = df_features['feature_name'].str.lower().str.replace(' ', '_')

                # Map to canonical names using reverse mapping
                df_features['canonical_name'] = df_features['normalized_name'].map(self.reverse_mapping)

                # Keep only recognized features, drop others silently
                df_features = df_features.dropna(subset=['canonical_name'])

                # Now pivot using canonical_name
                pivoted = df_features.pivot(index='product_id', columns='canonical_name', values='feature_value').reset_index()
                pivoted['product_id'] = pivoted['product_id'].astype(str)

                # Merge into main df
                df = df.merge(pivoted, left_on='id', right_on='product_id', how='left')
                df = df.drop(columns=['product_id'], errors='ignore')

                # If product_features supplied a room_type, use it to OVERRIDE the category-based room_type
                if 'room_type' in df.columns:
                    # 'room_type' column now has the feature value where available; keep category value as fallback
                    # The category-based room_type was stored as the 'room_type' from the df_categories merge
                    # After the pivot merge, pandas may suffix it.  We handle both cases:
                    if 'room_type_x' in df.columns and 'room_type_y' in df.columns:
                        df['room_type'] = df['room_type_y'].where(df['room_type_y'].notna(), df['room_type_x'])
                        df = df.drop(columns=['room_type_x', 'room_type_y'], errors='ignore')

            # === IMAGE HANDLING (unchanged) ===
            if not df_images.empty:
                df_images = df_images.drop_duplicates(subset=['product_id'], keep='first')
                df = df.merge(df_images, left_on='id', right_on='product_id', how='left')
                df['image'] = df['image_path']
                df = df.drop(columns=['image_path', 'product_id'], errors='ignore')
            else:
                df['image'] = 'default.jpg'

            df['id'] = df['id'].astype(str)

            # === BUDGET RANGE  ===
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
                df[col] = df[col].fillna('Unknown')

            # Title-case room_size and style; keep full title-case for room_type
            df['room_size'] = df['room_size'].str.capitalize()
            df['style'] = df['style'].str.capitalize()
            df['room_type'] = df['room_type'].apply(
                lambda v: v.title() if v and v != 'Unknown' else v
            )

            # === CATEGORY KEYWORD INFERENCE ===
            # Products whose room_type is 'Unknown' (or whose category is not a room-type name)
            # get a room_type inferred from their category name using keyword matching.
            def infer_room_type_from_category(row):
                """Also used for products whose category name is NOT itself a room-type label."""
                rt = str(row.get('room_type', '')).strip()
                cat_lower = rt.lower()
                # Check against keyword mapping
                for keywords, room_type_label in self.CATEGORY_ROOM_TYPE_MAP:
                    if any(kw in cat_lower for kw in keywords):
                        return room_type_label
                return rt  # return as-is (could be 'Unknown' or an exact match)

            df['room_type'] = df.apply(infer_room_type_from_category, axis=1)

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

        max_price = self.df['price'].max()
        if max_price and max_price > 0:
            price_norm = (self.df['price'] / max_price).values.reshape(-1, 1)
        else:
            price_norm = np.zeros((len(self.df), 1))
        price_weighted = np.repeat(price_norm, 3, axis=1)

        self.feature_matrix = np.hstack((encoded, price_weighted))

    def _normalize_value(self, value):
        if value is None:
            return ""
        normalized = str(value).strip().lower().replace('-', ' ').replace('_', ' ')
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _to_display_value(self, value, fallback="Unknown"):
        normalized = self._normalize_value(value)
        if not normalized:
            return fallback
        return normalized.title()

    def _compute_match_score(self, row, room_size, room_type, style, budget_range):
        comparisons = [
            (self._normalize_value(row.get('room_size')), room_size),
            (self._normalize_value(row.get('room_type')), room_type),
            (self._normalize_value(row.get('style')), style),
            (self._normalize_value(row.get('budget_range')), budget_range),
        ]
        matched = sum(1 for actual, expected in comparisons if actual == expected)
        return (matched / 4.0) * 100.0

    def search(self, room_size, room_type, style, budget_range, limit=5):
        room_size_norm = self._normalize_value(room_size)
        room_type_norm = self._normalize_value(room_type)
        style_norm = self._normalize_value(style)
        budget_norm = self._normalize_value(budget_range)

        room_size_display = self._to_display_value(room_size)
        room_type_display = self._to_display_value(room_type)
        style_display = self._to_display_value(style)
        budget_display = self._to_display_value(budget_range)

        scored_df = self.df.copy()
        scored_df['room_size_norm'] = scored_df['room_size'].apply(self._normalize_value)
        scored_df['room_type_norm'] = scored_df['room_type'].apply(self._normalize_value)
        scored_df['style_norm'] = scored_df['style'].apply(self._normalize_value)
        scored_df['budget_norm'] = scored_df['budget_range'].apply(self._normalize_value)

        exact_df = scored_df[
            (scored_df['room_size_norm'] == room_size_norm) &
            (scored_df['room_type_norm'] == room_type_norm) &
            (scored_df['style_norm'] == style_norm) &
            (scored_df['budget_norm'] == budget_norm)
        ].copy()

        has_exact_match = not exact_df.empty

        if has_exact_match:
            result_df = exact_df.head(limit).copy()
            result_df['match_score'] = 100.0
        else:
            # Style and budget are ALWAYS locked — never show wrong-style or
            # wrong-budget products. Only room_size is relaxed in fallback.
            # If no match exists for this room_type + style + budget, return empty.
            fallback_df = scored_df[
                (scored_df['room_type_norm'] == room_type_norm) &
                (scored_df['style_norm'] == style_norm) &
                (scored_df['budget_norm'] == budget_norm)
            ].copy()

            if fallback_df.empty:
                result_df = fallback_df
            else:
                fallback_df['match_score'] = fallback_df.apply(
                    lambda row: self._compute_match_score(
                        row,
                        room_size_norm,
                        room_type_norm,
                        style_norm,
                        budget_norm
                    ),
                    axis=1
                )

                result_df = fallback_df.sort_values(
                    by=['match_score', 'price'],
                    ascending=[False, True]
                ).head(limit).copy()

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

        room_size_adjusted = (not has_exact_match) and bool(suggestions)

        if has_exact_match:
            description = (
                f"Exact-match recommendations for your {room_size_display.lower()} {room_type_display.lower()} "
                f"in {style_display.lower()} style ({budget_display.lower()} budget)."
            )
        elif suggestions:
            description = (
                f"No exact match for your {room_size_display.lower()} room size, "
                f"but here are similar {style_display.lower()} {room_type_display.lower()} products "
                f"within your {budget_display.lower()} budget."
            )
        else:
            description = (
                f"No products found for your {room_size_display.lower()} {room_type_display.lower()} "
                f"in {style_display.lower()} style with a {budget_display.lower()} budget. "
                f"Try adjusting your room size or style to see more options."
            )

        return {
            'title': f"{style_display} {room_type_display} Design",
            'description': description,
            'suggestions': suggestions,
            'room_size_adjusted': room_size_adjusted,
            'tips': self._generate_design_tips(room_size_display, room_type_display, style_display, budget_display),
            'tags': self._generate_tags(room_size_display, room_type_display, style_display, budget_display)
        }

    def _generate_design_tips(self, room_size, room_type, style, budget_range):
        tips = []
        if room_size == "Small":
            tips += [
                "Choose one hero piece with hidden storage",
                "Float furniture on slim legs to show more floor",
                "Use wall-mounted shelves and vertical lamps to free walking space"
            ]
        elif room_size == "Medium":
            tips += [
                "Build a 3-point layout with a sofa, accent chair, and sideboard for balance",
                "Layer one large rug with a smaller texture rug for designer depth",
                "Use one statement light to define the room mood"
            ]
        else:
            tips += [
                "Create two mini-zones such as conversation and reading with matching color accents",
                "Use oversized art or tall shelving to avoid empty wall feel",
                "Mix materials like wood, fabric, and metal to add warmth at scale"
            ]

        if style == "Modern":
            tips += [
                "Pick low-profile furniture and one sculptural chair",
                "Use black metal details to sharpen clean lines",
                "Add one bold color object per zone, not per item"
            ]
        elif style == "Classic":
            tips += [
                "Anchor with timeless wood pieces and curved silhouettes",
                "Use symmetry with pairs such as lamps or chairs for a premium feel",
                "Layer rich textiles like velvet and linen for elegance"
            ]
        else:
            tips += [
                "Keep only functional pieces, then add one quiet-luxury texture",
                "Hide clutter with closed storage fronts",
                "Use tonal neutrals and natural light as decor"
            ]

        if budget_range == "Low":
            tips += [
                "Buy the frame now, then style with affordable cushions and throws",
                "Mix one new key item with refurbished vintage finds"
            ]
        elif budget_range == "Medium":
            tips += [
                "Invest in high-touch pieces such as the sofa or bed, then save on accessories",
                "Choose modular furniture so the setup can grow later"
            ]
        else:
            tips += [
                "Prioritize craftsmanship and timeless forms over trends",
                "Use custom sizing or finishes for a showroom-quality look"
            ]

        return tips

    def _generate_tags(self, room_size, room_type, style, budget_range):
        base = [room_size, room_type, style, f"{budget_range} Budget"]
        extras = {"Modern": ["Contemporary", "Sleek"], "Classic": ["Elegant", "Timeless"], "Minimalist": ["Clean", "Simple"]}.get(style, [])[:2]
        room_extras = {"Living Room": ["Cozy", "Entertainment"], "Bedroom": ["Restful", "Relaxation"], "Dining": ["Gathering", "Meals"]}.get(room_type, [])[:2]
        return base + extras + room_extras

    def get_product_by_id(self, product_id):
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
        product_id = str(product_id)
        product_row = self.df[self.df['id'] == product_id]

        if product_row.empty:
            return []

        target_idx = product_row.index[0]
        target_room_type_norm = self._normalize_value(product_row.iloc[0].get('room_type'))

        # Content-based similarity over structured attributes
        categorical_cols = ['room_size', 'room_type', 'style', 'budget_range']
        encoded_features = self.encoder.transform(self.df[categorical_cols])
        target_content_vector = encoded_features[target_idx].reshape(1, -1)
        content_similarity = cosine_similarity(target_content_vector, encoded_features)[0]

        # Price-derived similarity (normalized distance on [0,1])
        max_price = self.df['price'].max()
        if max_price and max_price > 0:
            normalized_prices = self.df['price'] / max_price
        else:
            normalized_prices = pd.Series(np.zeros(len(self.df)), index=self.df.index)

        target_price = normalized_prices.iloc[target_idx]
        price_similarity = 1 - np.abs(normalized_prices - target_price)
        price_similarity = np.clip(price_similarity, 0, 1)

        # Weighted final score (report-friendly and interpretable)
        content_weight = 0.8
        price_weight = 0.2
        similarities = (content_weight * content_similarity) + (price_weight * price_similarity)
        similarities[target_idx] = -1  # Exclude itself

        candidates = self.df.copy()
        candidates['similarity'] = similarities
        candidates = candidates[candidates['similarity'] >= 0].copy()
        candidates['room_type_norm'] = candidates['room_type'].apply(self._normalize_value)

        if target_room_type_norm and target_room_type_norm != 'unknown':
            same_room_type = candidates[candidates['room_type_norm'] == target_room_type_norm]
            other_room_types = candidates[candidates['room_type_norm'] != target_room_type_norm]

            same_room_type = same_room_type.sort_values(by=['similarity', 'price'], ascending=[False, True])
            other_room_types = other_room_types.sort_values(by=['similarity', 'price'], ascending=[False, True])
            ranked_candidates = pd.concat([same_room_type, other_room_types], axis=0)
        else:
            ranked_candidates = candidates.sort_values(by=['similarity', 'price'], ascending=[False, True])

        result_df = ranked_candidates.head(limit).copy()

        related_products = [
            {
                'id': row['id'],
                'name': row['name'],
                'price': float(row['price']),
                'description': row['description'],
                'image': row['image'],
                'link': f"/product/{row['id']}",
                'match_score': round(row['similarity'] * 100, 2)
            }
            for _, row in result_df.iterrows()
        ]

        return related_products

    def get_available_filters(self):
        self.df = self._load_dataset()
        self._prepare_similarity_features()
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