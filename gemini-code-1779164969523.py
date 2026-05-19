import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib_fontja
import trimesh

# ==========================================
# 1. 関数定義
# ==========================================

def load_stl_meshes(stl_files):
    """アップロードされたSTLファイルをtrimeshオブジェクトとして読み込む"""
    meshes = {}
    logs = []
    for stl_file in stl_files:
        try:
            room_name = os.path.splitext(stl_file.name)[0]
            # trimeshでSTLを読み込み
            mesh = trimesh.load(file_obj=stl_file, file_type='stl')
            meshes[room_name] = mesh
        except Exception as e:
            logs.append(f"❌ STL読み込みエラー ({stl_file.name}): {e}")
    return meshes, logs

def detect_rooms_from_coords(df, meshes, offset_dist=0.05):
    """座標から開口部の軸と、隣接する2つの部屋を判定し、詳細なデバッグログを返す"""
    if not all(col in df.columns for col in ['X[m]', 'Y[m]', 'Z[m]']):
        return None, None, None, "座標列なし", []

    cx = df['X[m]'].mean()
    cy = df['Y[m]'].mean()
    cz = df['Z[m]'].mean()

    stds = {'x': df['X[m]'].std(), 'y': df['Y[m]'].std(), 'z': df['Z[m]'].std()}
    detected_axis = min(stds, key=stds.get)

    pt_plus = [cx, cy, cz]
    pt_minus = [cx, cy, cz]
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[detected_axis]
    pt_plus[axis_idx] += offset_dist
    pt_minus[axis_idx] -= offset_dist

    debug_logs = []
    debug_logs.append(f"📍 中心:{cx:.3f}, {cy:.3f}, {cz:.3f} | 軸:{detected_axis}")
    debug_logs.append(f"   ➕Plus点 : {pt_plus[0]:.3f}, {pt_plus[1]:.3f}, {pt_plus[2]:.3f}")
    debug_logs.append(f"   ➖Minus点: {pt_minus[0]:.3f}, {pt_minus[1]:.3f}, {pt_minus[2]:.3f}")

    def is_inside(mesh, pt, point_name):
        min_b, max_b = mesh.bounds
        in_box = (min_b[0] <= pt[0] <= max_b[0] and
                  min_b[1] <= pt[1] <= max_b[1] and
                  min_b[2] <= pt[2] <= max_b[2])
        if not in_box:
            return False, "Box外"
            
        try:
            ray_origins = np.array([pt])
            ray_directions = np.array([[0.0, 0.0, 1.0]]) # 真上にレーザーを撃つ
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions
            )
            hits = len(locations)
            hit_zs = [round(loc[2], 3) for loc in locations]
            
            result = (hits % 2 == 1)
            msg = f"Box内 ➔ レーザー貫通 {hits}回 (交差点Z: {hit_zs})"
            return result, msg
            
        except Exception as e:
            return False, f"計算エラー: {e}"

    room_plus = "外気(未定義)"
    room_minus = "外気(未定義)"

    sorted_meshes = sorted(meshes.items(), key=lambda item: item[1].bounding_box.volume)

    for room_name, mesh in sorted_meshes:
        if is_inside(mesh, pt_plus):
            room_plus = room_name
            break
            
    for room_name, mesh in sorted_meshes:
        if is_inside(mesh, pt_minus):
            room_minus = room_name
            break

    return detected_axis, room_plus, room_minus, None

def process_cfd_files(stl_files, cfd_files, rho, cp, lv, threshold, calc_latent, hum_col):
    meshes, logs = load_stl_meshes(stl_files)
    if not meshes:
        logs.append("⚠️ 有効な3Dメッシュ(STL)が読み込めませんでした。")
        return None, None, None, logs
# ==========================================
    # 🔍 【デバッグ】STLの読み込み状況と座標スケールを確認
    # ==========================================
    st.markdown("### 🔍 【デバッグ】STL読み込み状況")
    for room_name, mesh in meshes.items():
        # mesh.bounds は [[X最小, Y最小, Z最小], [X最大, Y最大, Z最大]] を返します
        min_bound = np.round(mesh.bounds[0], 2)
        max_bound = np.round(mesh.bounds[1], 2)
        st.write(f"- **{room_name}**: X範囲[{min_bound[0]} 〜 {max_bound[0]}], "
                 f"Y範囲[{min_bound[1]} 〜 {max_bound[1]}], "
                 f"Z範囲[{min_bound[2]} 〜 {max_bound[2]}]")
    st.markdown("---")
    # ==========================================
    opening_results_list = []
    
    total_files = len(cfd_files)
    progress_bar = st.progress(0)

    for i, uploaded_file in enumerate(cfd_files):
        progress_bar.progress((i + 1) / total_files)
        file_name = uploaded_file.name
        file_key = os.path.splitext(file_name)[0]

        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='cp932')
            
           # --- (A) 空間判定 ---
            # 戻り値に debug_logs を追加で受け取る
            axis, r_plus, r_minus, err, debug_logs = detect_rooms_from_coords(df, meshes, offset_dist=0.05)

            if err:
                logs.append(f"⚠️ {file_name}: {err}")
                continue
                
            # --- (B) 熱量計算 ---
            flow_col, temp_col = '流量[m3/h]', 'スカラー量[℃]'
            df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
            df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
            df.dropna(subset=[flow_col, temp_col], inplace=True)

            # 顕熱計算 (W)
            df['heat_sensible_W'] = df[flow_col] * rho * cp * df[temp_col] * 1000 / 3600
            net_sensible = df['heat_sensible_W'].sum()
            
            # 潜熱計算 (W)
            net_latent = 0.0
            if calc_latent and hum_col in df.columns:
                df[hum_col] = pd.to_numeric(df[hum_col], errors='coerce')
                df.dropna(subset=[hum_col], inplace=True)
                # 潜熱 = 流量 * 密度 * 蒸発潜熱 * 絶対湿度
                df['heat_latent_W'] = df[flow_col] * rho * lv * df[hum_col] * 1000 / 3600
                net_latent = df['heat_latent_W'].sum()
            elif calc_latent:
                logs.append(f"⚠️ {file_name}: 湿度列 '{hum_col}' が見つからないため潜熱を0とします")

            net_total = net_sensible + net_latent
            
            # 流量計算
            gross_positive_flow = df[df[flow_col] > 0][flow_col].sum()
            gross_negative_flow = df[df[flow_col] < 0][flow_col].sum()

            opening_results_list.append({
                '開口部': file_key,
                '方向': axis,
                'Plus_Room': r_plus,
                'Minus_Room': r_minus,
                '総プラス流量[m3/h]': gross_positive_flow,
                '総マイナス流量[m3/h]': gross_negative_flow,
                '顕熱移動量[W]': net_sensible,
                '潜熱移動量[W]': net_latent,
                '全熱移動量[W]': net_total
            })

        except Exception as e:
            logs.append(f"❌ 計算エラー: {file_name} ({e})")

    if not opening_results_list:
        return None, None, None, logs
    
    results_df = pd.DataFrame(opening_results_list)

    # --- 集計処理 ---
    heat_movements = []
    # 移動熱量(全熱)がプラス
    df_heat_pos = results_df[results_df['全熱移動量[W]'] > 0]
    heat_movements.append(pd.DataFrame({'室名': df_heat_pos['Minus_Room'], '方向': '流出', '熱量[W]': df_heat_pos['全熱移動量[W]']}))
    heat_movements.append(pd.DataFrame({'室名': df_heat_pos['Plus_Room'], '方向': '流入', '熱量[W]': df_heat_pos['全熱移動量[W]']}))
    # 移動熱量(全熱)がマイナス
    df_heat_neg = results_df[results_df['全熱移動量[W]'] < 0]
    heat_movements.append(pd.DataFrame({'室名': df_heat_neg['Plus_Room'], '方向': '流出', '熱量[W]': df_heat_neg['全熱移動量[W]'].abs()}))
    heat_movements.append(pd.DataFrame({'室名': df_heat_neg['Minus_Room'], '方向': '流入', '熱量[W]': df_heat_neg['全熱移動量[W]'].abs()}))
    
    if heat_movements:
        heat_df = pd.concat(heat_movements).groupby(['室名', '方向'])['熱量[W]'].sum().unstack(fill_value=0)
        room_heat_summary_df = pd.DataFrame({
            '総流出熱量[W]': heat_df.get('流出', 0),
            '総流入熱量[W]': heat_df.get('流入', 0),
            '処理熱量[W]': heat_df.get('流出', 0) - heat_df.get('流入', 0)
        }).reset_index()
    else:
        room_heat_summary_df = pd.DataFrame()

    # 風量収支集計
    flow_movements = []
    flow_movements.append(pd.DataFrame({'室名': results_df['Minus_Room'], '方向': '流出', '流量[m3/h]': results_df['総プラス流量[m3/h]']}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Plus_Room'], '方向': '流入', '流量[m3/h]': results_df['総プラス流量[m3/h]']}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Plus_Room'], '方向': '流出', '流量[m3/h]': results_df['総マイナス流量[m3/h]'].abs()}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Minus_Room'], '方向': '流入', '流量[m3/h]': results_df['総マイナス流量[m3/h]'].abs()}))

    if flow_movements:
        flow_df = pd.concat(flow_movements).groupby(['室名', '方向'])['流量[m3/h]'].sum().unstack(fill_value=0)
        room_flow_summary_df = pd.DataFrame({
            '総流出流量[m3/h]': flow_df.get('流出', 0),
            '総流入流量[m3/h]': flow_df.get('流入', 0),
            '風量収支[m3/h]': flow_df.get('流出', 0) - flow_df.get('流入', 0)
        }).reset_index()
    else:
        room_flow_summary_df = pd.DataFrame()

    return results_df, room_heat_summary_df, room_flow_summary_df, logs

# (※ create_heat_chart関数は以前と同じため割愛・そのまま使用可能です)
def create_heat_chart(room_heat_summary_df, fig_width, fig_height, font_size, y_max, custom_colors, show_legend, category_map, mode):
    # --- データ準備 ---
    if "暖房" in mode:
        label_passive = "各室熱損失"
        label_active = "投入熱量"
        passive = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] < 0].set_index('室名')['処理熱量[W]'].abs()
        active = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] > 0].set_index('室名')['処理熱量[W]']
    else: 
        label_passive = "各室負荷"
        label_active = "処理熱量"
        passive = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] > 0].set_index('室名')['処理熱量[W]']
        active = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] < 0].set_index('室名')['処理熱量[W]'].abs()
        
    plot_df_base = pd.DataFrame({label_passive: passive , label_active: active}).T.fillna(0)

    # 並べ替えロジック
    desired_order = []
    for rooms in category_map.values():
        desired_order.extend(rooms)
    
    current_columns = plot_df_base.columns.tolist()
    ordered_columns = [col for col in desired_order if col in current_columns]
    remaining_columns = [col for col in current_columns if col not in desired_order]
    final_column_order = ordered_columns + remaining_columns
    
    plot_df = plot_df_base[final_column_order]
    
    # 色の適用
    colors = [custom_colors.get(room, '#AAAAAA') for room in final_column_order]

    # 描画
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8, legend=False)

    # 見た目調整
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='#cccccc')
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(axis='y', length=0, labelsize=font_size)
    ax.tick_params(axis='x', length=0)
    plt.xticks(rotation=0, fontsize=font_size)
    plt.ylabel('全処理熱量[W]', fontsize=font_size)
    
    if y_max > 0:
        ax.set_ylim(0, y_max)
    plt.axhline(0, color='black', linewidth=0.8)

    # バーの数値ラベル
    for i, container in enumerate(ax.containers):
        labels = [f"{v:,.0f}" if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', color='black', fontsize=font_size*0.8, fontweight='bold')

    # 凡例の作成
    if show_legend:
        handles, labels_legend = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        dummy_handle = mpatches.Patch(visible=False)

        for category_name, rooms_in_category in reversed(category_map.items()):
            category_handles_labels = []
            for room_name in reversed(rooms_in_category):
                if room_name in labels_legend:
                    index = labels_legend.index(room_name)
                    category_handles_labels.append((handles[index], f"  {room_name}"))
            
            if category_handles_labels:
                if category_name:
                    new_handles.append(dummy_handle)
                    new_labels.append(f"--- {category_name} ---")
                for handle, label in category_handles_labels:
                    new_handles.append(handle)
                    new_labels.append(label)

        remaining_items = [(handles[i], f"  {labels_legend[i]}") for i, label in enumerate(labels_legend) if label not in desired_order]
        if remaining_items:
            new_handles.append(dummy_handle)
            new_labels.append("▼ 未分類")
            for handle, label in remaining_items:
                new_handles.append(handle)
                new_labels.append(label)

        ax.legend(handles=new_handles, labels=new_labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=font_size*0.9)

    total_pos = passive.sum()
    total_neg = active.sum()
    
    return fig, total_pos, total_neg

# ==========================================
# 2. アプリケーション UI
# ==========================================

st.set_page_config(page_title="CFD 熱量分析ツール (BIM連動版)", layout="wide")
st.title("CFD 全熱量分配 & 風量バランス分析 (自動領域判定)")
st.markdown("Grasshopperから書き出した部屋のSTLモデルを利用し、CSV座標から隣接室を自動判定します。")

if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0
        
with st.sidebar:
    st.header("1. 解析設定")
    mode = st.radio("モード", ["冷房", "暖房"])
    
    st.header("2. 熱力学定数")
    rho = st.number_input("空気密度 ρ [kg/m3]", value=1.20)
    cp = st.number_input("空気比熱 Cp [J/g・K]", value=1.006, format="%.3f")
    
    st.divider()
    calc_latent = st.checkbox("潜熱（湿度）も計算する", value=True)
    if calc_latent:
        lv = st.number_input("水の蒸発潜熱 Lv [kJ/kg]", value=2501.0, format="%.1f")
        hum_col = st.text_input("CSV内の絶対湿度列名", value="スカラー量[kg/kg(DA)]")
    else:
        lv = 0.0
        hum_col = ""
        
    threshold = st.number_input("風量収支許容誤差 [m3/h]", value=1.0)
    
    st.header("3. ファイルアップロード")
    st.info("部屋の領域を示す3Dモデル (複数選択)")
    stl_files = st.file_uploader("部屋のSTLファイル群 (.stl)", type="stl", accept_multiple_files=True)
    
    st.info("FDで書きだした開口部のCSV (複数選択)")
    cfd_files = st.file_uploader(
        "CFD解析結果CSV",
        type="csv",
        accept_multiple_files=True,
        key=f"cfd_uploader_{st.session_state['uploader_key']}"
    )

    def reset_files():
        st.session_state['uploader_key'] += 1
        st.session_state['analyzed'] = False

    if st.button("リセット"):
        reset_files()
        st.rerun()

# --- メイン処理 ---
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
    st.session_state['results_df'] = None
    st.session_state['room_heat_df'] = None
    st.session_state['room_flow_df'] = None
    st.session_state['logs'] = []

if st.button("解析実行", type="primary"):
    if not stl_files or not cfd_files:
        st.warning("STLファイルとCSVファイルの両方をアップロードしてください。")
    else:
        with st.spinner("空間判定と熱量計算を実行中..."):
            results_df, room_heat_df, room_flow_df, logs = process_cfd_files(
                stl_files, cfd_files, rho, cp, lv, threshold, calc_latent, hum_col
            )
            
            if results_df is not None:
                st.session_state['results_df'] = results_df
                st.session_state['room_heat_df'] = room_heat_df
                st.session_state['room_flow_df'] = room_flow_df
                st.session_state['logs'] = logs
                st.session_state['analyzed'] = True
                st.success("解析完了！")
            else:
                st.session_state['logs'] = logs
                st.error("有効なデータが作成されませんでした。ログを確認してください。")

if st.session_state['analyzed']:
    results_df = st.session_state['results_df']
    room_heat_df = st.session_state['room_heat_df']
    room_flow_df = st.session_state['room_flow_df']
    logs = st.session_state['logs']

    with st.expander("エラー・警告ログ", expanded=False):
        for log in logs:
            if "❌" in log: st.error(log)
            elif "⚠️" in log: st.warning(log)
            else: st.info(log)

    tab1, tab2, tab3 = st.tabs(["風量収支チェック", "全熱量分配グラフ", "計算詳細"])

    # --- Tab 1: 風量バランス ---
    with tab1:
        st.subheader("風量収支チェック")
        warning_count = 0
        for index, row in room_flow_df.iterrows():
            room = row['室名']
            balance = row['風量収支[m3/h]']
            if balance > threshold:
                st.error(f"⚠️ {room}: 流出過多 (流入不足) +{balance:.2f} m3/h")
                warning_count += 1
            elif balance < -threshold:
                st.error(f"⚠️ {room}: 流入過多 (流出不足) {balance:.2f} m3/h")
                warning_count += 1
            else:
                st.success(f"{room}: OK ({balance:+.2f} m3/h)")
        if warning_count == 0:
            st.info("✅ 全室で風量収支が許容値以下です。")

    # --- Tab 2: グラフ ---
    with tab2:
        st.subheader("各室およびエアコンの空調処理熱量 (全熱)")
        all_rooms = sorted(room_heat_df['室名'].unique())

        with st.expander("グラフをカスタマイズする", expanded=False):
            # (以前と同じカスタマイズUIのためコード省略・実装時はそのまま組み込んでください)
            st.markdown("※ UIコードは以前のものと同一です")
            
            # デフォルト設定 (仮)
            custom_category_map = {"全室": all_rooms}
            default_colors = {"LDK": "#FF7F50", "エアコン内部": "#87CEEB"}
            
            fig_w, fig_h, font_size, y_max, show_legend = 6.0, 8.0, 14, 0, True

        try:
            fig, total_passive, total_active = create_heat_chart(
                room_heat_df, fig_w, fig_h, font_size, y_max, default_colors, show_legend, custom_category_map, mode
            )
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            label_left = "各室熱損失合計" if "暖房" in mode else "各室熱負荷合計"
            label_right = "投入熱量" if "暖房" in mode else "処理熱量"
            col1.metric(label_left, f"{total_passive:,.1f} W")
            col2.metric(label_right, f"{total_active:,.1f} W")
        except Exception as e:
            st.error(f"グラフ作成エラー: {e}")

    # --- Tab 3: 計算詳細 ---
    with tab3:
        st.markdown("### 📥 データダウンロード")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        col_dl1.download_button("表1 (開口部詳細)", results_df.to_csv(index=False).encode('shift_jis'), "results_raw.csv")
        col_dl2.download_button("表2 (熱量集計)", room_heat_df.to_csv(index=False).encode('shift_jis'), "results_heat.csv")
        col_dl3.download_button("表3 (風量収支)", room_flow_df.to_csv(index=False).encode('shift_jis'), "results_flow.csv")
        st.divider()

        st.markdown("### (表1) 開口部別 詳細データ")
        st.dataframe(results_df)
