# -*- coding: utf-8 -*-
"""
VA Label Loader Class (supports MemoMusic and IADS-E)
"""

import pandas as pd


class VALoader:
    def __init__(self, excel_path):
        self.data = pd.read_excel(excel_path, sheet_name=None)


    def get_memo_va(self, sub_id: str) -> dict:
        """
        Get all music labels of the specified subject in the MemoMusic table, and store the data according to the experiment round (exp_num).
        Return format:
        {
            sub_id: {  # Subject ID
                exp_num: {  # Experiment round
                    music_id: {"V_after": ..., "A_after": ..., "V_music": ..., "A_music": ..., ...},
                    ...
                }
            }
        }
        """
        df = self.data["MemoMusic"]
        df["user_ID"] = df["user_ID"].astype(str).str.zfill(2)
        sub_id = sub_id.split('_')[1].zfill(2)

        df_sub = df[df["user_ID"] == sub_id]

        va_dict = {}
        # Iterate over each row to get exp_num and save music labels
        for _, row in df_sub.iterrows():
            exp_num = 'exp_' + str(row["exp_num"])  # Experiment round

            if exp_num not in va_dict:
                va_dict[exp_num] = {}

            music_id = 'music_' + str(row["music_num"]).zfill(2)  # Get music ID
            va_dict[exp_num][music_id] = {
                "V_after": int(row["V_after"]),
                "A_after": int(row["A_after"]),
                "V_before": int(row["V_before"]),
                "A_before": int(row["A_before"]),
                "V_distance": int(row["V_Distance"]),
                "A_distance": int(row["A_Distance"]),
                "V_music": int(row["V_music"]),
                "A_music": int(row["A_music"])
            }
        return va_dict

    def get_iads_va(self, sub_id: str) -> dict:
        """
        Get all music labels of the specified subject in the IADS-E table, and store the data according to the quadrant information (session_num).
        Return format:
        {
            sub_id: {  # Subject ID
                session_num: {  # Quadrant information
                    music_id: {"Valence": ..., "Arousal": ...},
                    ...
                             }
                    }
        }
        """
        df = self.data["IADS-E"]
        df["user_ID"] = df["user_ID"].astype(str).str.zfill(2)
        sub_id = sub_id.split('_')[1].zfill(2)

        df_sub = df[df["user_ID"] == sub_id]

        va_dict = {}
        for _, row in df_sub.iterrows():
            Q_num = 'Q' + str(row["session_num"])  # Quadrant information

            if Q_num not in va_dict:
                va_dict[Q_num] = {}

            music_id = 'music_' + str(row["music_num"]).zfill(2)  # Get music ID
            va_dict[Q_num][music_id] = {
                "Valence": int(row["Valence"]),
                "Arousal": int(row["Arousal"]),
                "music_V": float(row["music_V"]),
                "music_A": float(row["music_A"])
            }
        return va_dict