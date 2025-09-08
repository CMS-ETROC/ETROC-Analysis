## --------------- Deprecated -----------------------
## --------------------------------------
# def sort_filter(group):
#     return group.sort_values(by=['board'], ascending=True)

## --------------------------------------
# def distance_filter(group, distance):
#     board0_row = group[(group["board"] == 0)]
#     board3_row = group[(group["board"] == 3)]
#     board0_col = group[(group["board"] == 0)]
#     board3_col = group[(group["board"] == 3)]

#     if not board0_row.empty and not board3_row.empty and not board0_col.empty and not board3_col.empty:
#         row_index_diff = abs(board0_row["row"].values[0] - board3_row["row"].values[0])
#         col_index_diff = abs(board0_col["col"].values[0] - board3_col["col"].values[0])
#         return row_index_diff < distance and col_index_diff < distance
#     else:
#         return False

## --------------------------------------
# def event_display_withPandas(
#         input_df: pd.DataFrame,
#     ):
#     # Loop over unique evt values
#     unique_evts = input_df['evt'].unique()

#     for cnt, evt in enumerate(unique_evts):
#         if cnt > 15: break

#         selected_subset_df = input_df[input_df['evt'] == evt]

#         fig = plt.figure()
#         ax1 = fig.add_subplot(111, projection='3d')
#         ax1.grid(False)

#         # Create a meshgrid for the contourf
#         xx, yy = np.meshgrid(np.arange(16), np.arange(16))

#         for board_value in [0, 3]:
#             board_subset_df = selected_subset_df[selected_subset_df['board'] == board_value]

#             # Create a 16x16 binary grid with 1s where "cal" exists, 0s otherwise
#             cal_grid = np.zeros((16, 16))
#             for _, row in board_subset_df.iterrows():
#                 cal_grid[row['row'], row['col']] = 1

#             # Plot the contourf for this board value
#             ax1.contourf(xx, yy, cal_grid, 100, zdir='z', offset=board_value, alpha=0.15, cmap="plasma")

#         ax1.set_zlim((0., 3.0))  # Adjust z-axis limit based on your board values
#         ax1.set_xlabel('COL', fontsize=15, labelpad=15)
#         ax1.set_ylabel('ROW', fontsize=15, labelpad=15)
#         ax1.invert_xaxis()
#         ax1.invert_yaxis()
#         ticks = range(0, 16)
#         ax1.set_xticks(ticks)
#         ax1.set_yticks(ticks)
#         ax1.set_xticks(ticks=range(16), labels=[], minor=True)
#         ax1.set_yticks(ticks=range(16), labels=[], minor=True)
#         ax1.set_zticks(ticks=[0, 1, 3], labels=["Bottom", "Middle", "Top"])
#         ax1.tick_params(axis='x', labelsize=8)  # You can adjust the 'pad' value
#         ax1.tick_params(axis='y', labelsize=8)
#         ax1.tick_params(axis='z', labelsize=8)
#         ax1.grid(visible=False, axis='z')
#         ax1.grid(visible=True, which='major', axis='x')
#         ax1.grid(visible=True, which='major', axis='y')
#         plt.title(f'Event {evt}')

#         del xx, yy, fig, ax1  # Clear the figure to avoid overlapping plots

## --------------------------------------
# def find_maximum_event_combination(
#         input_df: pd.DataFrame,
#         board_pixel_info: list,
#     ):
#     # Step 1: Filter the rows where board is 1, col is 6, and row is 15
#     selected_rows = input_df[(input_df['board'] == board_pixel_info[0]) & (input_df['row'] == board_pixel_info[1]) & (input_df['col'] == board_pixel_info[2])]

#     # Step 2: Get the unique "evt" values from the selected rows
#     unique_evts = selected_rows['evt'].unique()

#     # Step 3: Filter rows where board is 0 or 3 and "evt" is in unique_evts
#     filtered_rows = input_df[(input_df['board'].isin([0, 3])) & (input_df['evt'].isin(unique_evts))]

#     result_df = pd.concat([selected_rows, filtered_rows], ignore_index=True)
#     result_df = result_df.sort_values(by="evt")
#     result_df.reset_index(inplace=True, drop=True)

#     test_group = result_df.groupby(['board', 'row', 'col'])
#     count_df = test_group.size().reset_index(name='count')

#     row0 = count_df.loc[count_df[count_df['board'] == 0]['count'].idxmax()]
#     row3 = count_df.loc[count_df[count_df['board'] == 3]['count'].idxmax()]

#     print(f"Board 0, Row: {row0['row']}, Col: {row0['col']}, Count: {row0['count']}")
#     print(f"Board 3, Row: {row3['row']}, Col: {row3['col']}, Count: {row3['count']}")

#     del selected_rows, unique_evts, filtered_rows, test_group, count_df, row0, row3
#     return result_df

## --------------------------------------
# def making_3d_heatmap_byPandas(
#         input_df: pd.DataFrame,
#         chipLabels: list,
#         figtitle: list,
#         figtitle_tag: str,
#     ):
#     # Create a 3D subplot

#     for idx, id in enumerate(chipLabels):

#         # Create the 2D heatmap for the current chip label
#         hits_count_by_col_row_board = input_df[input_df['board'] == int(id)].groupby(['col', 'row'])['evt'].count().reset_index()
#         hits_count_by_col_row_board = hits_count_by_col_row_board.rename(columns={'evt': 'hits'})
#         pivot_table = hits_count_by_col_row_board.pivot_table(index='row', columns='col', values='hits', fill_value=0)

#         if pivot_table.shape[1] != 16:
#             continue

#         fig = plt.figure(figsize=(15, 10))
#         ax = fig.add_subplot(111, projection='3d')

#         # Create a meshgrid for the 3D surface
#         x, y = np.meshgrid(np.arange(16), np.arange(16))
#         z = pivot_table.values
#         dx = dy = 0.75  # Width and depth of the bars

#         # Create a 3D surface plot
#         ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z).flatten(), dx, dy, z.flatten(), shade=True)

#         # Customize the 3D plot settings as needed
#         ax.set_xlabel('COL', fontsize=15, labelpad=15)
#         ax.set_ylabel('ROW', fontsize=15, labelpad=15)
#         ax.set_zlabel('Hits', fontsize=15, labelpad=-35)
#         ax.invert_xaxis()
#         ticks = range(0, 16)
#         ax.set_xticks(ticks)
#         ax.set_yticks(ticks)
#         ax.set_xticks(ticks=range(16), labels=[], minor=True)
#         ax.set_yticks(ticks=range(16), labels=[], minor=True)
#         ax.tick_params(axis='x', labelsize=8)  # You can adjust the 'pad' value
#         ax.tick_params(axis='y', labelsize=8)
#         ax.tick_params(axis='z', labelsize=8)
#         ax.set_title(f"Heat map 3D {figtitle[idx]}", fontsize=16)
#         plt.tight_layout()

## --------------------------------------
# def making_scatter_with_plotly(
#         input_df: pd.DataFrame,
#         output_name: str,
#     ):
#     import plotly.express as px
#     fig = px.scatter_matrix(
#         input_df,
#         dimensions=input_df.columns,
#         # labels = labels,
#         # color=color_column,
#         # title = "Scatter plot comparing variables for each board<br><sup>Run: {}{}</sup>".format(run_name, extra_title),
#         opacity = 0.2,
#     )

#     ## Delete half of un-needed plots
#     fig.update_traces(
#         diagonal_visible = False,
#         showupperhalf=False,
#         marker = {'size': 3}
#     )

#     for k in range(len(fig.data)):
#         fig.data[k].update(
#             selected = dict(
#             marker = dict(
#                 #opacity = 1,
#                 #color = 'blue',
#                 )
#             ),
#             unselected = dict(
#                 marker = dict(
#                     #opacity = 0.1,
#                     color="grey"
#                     )
#                 ),
#             )

#     fig.write_html(
#         f'{output_name}.html',
#         full_html = False,
#         include_plotlyjs = 'cdn',
#     )

## --------------------------------------
# def draw_hist_plot_pull(
#     input_hist: hist.Hist,
#     fig_title: str,
# ):
#     fig = plt.figure(figsize=(15, 10))
#     grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
#     main_ax = fig.add_subplot(grid[0])
#     subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
#     plt.setp(main_ax.get_xticklabels(), visible=False)

#     main_ax_artists, sublot_ax_arists = input_hist.plot_pull(
#         "gaus",
#         eb_ecolor="steelblue",
#         eb_mfc="steelblue",
#         eb_mec="steelblue",
#         eb_fmt="o",
#         eb_ms=6,
#         eb_capsize=1,
#         eb_capthick=2,
#         eb_alpha=0.8,
#         fp_c="hotpink",
#         fp_ls="-",
#         fp_lw=2,
#         fp_alpha=0.8,
#         bar_fc="royalblue",
#         pp_num=3,
#         pp_fc="royalblue",
#         pp_alpha=0.618,
#         pp_ec=None,
#         ub_alpha=0.2,
#         fit_fmt= r"{name} = {value:.4g} $\pm$ {error:.4g}",
#         ax_dict= {"main_ax":main_ax,"pull_ax":subplot_ax},
#     )
#     hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)
#     main_ax.set_title(f'{fig_title}', loc="right", size=25)

## --------------------------------------
# def lmfit_gaussfit_with_pulls(
#         model: str,
#         input_data: np.array,
#         input_hist: hist.Hist,
#         width_factor: float,
#         fig_title: str,
#         chipNames: str,
#         use_pred_uncert: bool,
#         no_show_fit: bool,
#         no_draw: bool,
#         get_chisqure: bool = False,
#         full_data: bool = False,
#         save: bool = False,
#     ):

#     from lmfit.models import GaussianModel
#     # from lmfit.lineshapes import gaussian

#     mod = GaussianModel(nan_policy='omit')

#     if full_data:
#         fit_min = input_data.min()
#         fit_max = input_data.max()
#         input_hist_peak = input_hist
#         centers = input_hist.axes[0].centers
#     else:
#         fit_min = input_data.mean()-width_factor*input_data.std()
#         fit_max = input_data.mean()+width_factor*input_data.std()


#     input_hist_peak = input_hist[(fit_min)*1j:(fit_max)*1j]
#     centers = input_hist.axes[0].centers
#     fit_centers = input_hist_peak.axes[0].centers
#     # pars = mod.guess(input_hist.values(), x=centers)
#     pars = mod.guess(input_hist_peak.values(), x=fit_centers)
#     out = mod.fit(input_hist_peak.values(), pars, x=fit_centers, weights=1/np.sqrt(input_hist_peak.values()))

#     if not no_draw:

#         x_range = np.linspace(fit_min, fit_max, 500)
#         # x_range = np.linspace(centers[0], centers[-1], 500)
#         dely = out.eval_uncertainty(x=x_range)

#         # popt = [par for name, par in out.best_values.items()]
#         # pcov = out.covar

#         # if np.isfinite(pcov).all():
#         #     n_samples = 100
#         #     vopts = np.random.multivariate_normal(popt, pcov, n_samples)
#         #     # sampled_ydata = np.vstack([gaussian(fit_centers, *vopt).T for vopt in vopts])
#         #     # sampled_ydata = np.vstack([gaussian(np.linspace(fit_min-1, fit_max+1, 500), *vopt).T for vopt in vopts])
#         #     sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
#         #     dely = np.nanstd(sampled_ydata, axis=0)
#         #     # model_uncert = np.nanstd(sampled_ydata, axis=0)

#         # else:
#         #     # model_uncert = np.zeros_like(np.sqrt(input_hist.variances()))
#         #     dely = np.zeros_like(np.sqrt(input_hist.variances()))

#         ### Calculate pull
#         if use_pred_uncert:
#             pulls = (input_hist.values() - out.eval(x=centers))/np.sqrt(out.eval(x=centers))
#         else:
#             pulls = (input_hist.values() - out.eval(x=centers))/np.sqrt(input_hist.variances())
#         pulls[np.isnan(pulls) | np.isinf(pulls)] = 0

#         left_edge = centers[0]
#         right_edge = centers[-1]

#         # Pull: plot the pulls using Matplotlib bar method
#         width = (right_edge - left_edge) / len(pulls)

#         fig = plt.figure()

#         if no_show_fit:
#             grid = fig.add_gridspec(1, 1, hspace=0)
#             main_ax = fig.add_subplot(grid[0])
#             hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)
#             main_ax.set_title(f'{fig_title}', loc="right", size=20)
#             main_ax.errorbar(centers, input_hist.values(), np.sqrt(input_hist.variances()),
#                             ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
#                             ms=6, capsize=1, capthick=2, alpha=0.8)
#             main_ax.plot(x_range, out.eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
#                         label=fr"$\mu:{out.params['center'].value:.2f}, \sigma: {abs(out.params['sigma'].value):.2f}$")
#             main_ax.set_ylabel('Counts')
#             main_ax.set_ylim(-20, None)
#             main_ax.set_xlabel(r'Time Walk Corrected $\Delta$TOA [ns]')
#             plt.tight_layout()

#         else:
#             grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
#             main_ax = fig.add_subplot(grid[0])
#             subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
#             plt.setp(main_ax.get_xticklabels(), visible=False)
#             hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)

#             main_ax.set_title(f'{fig_title}', loc="right", size=25)
#             main_ax.errorbar(centers, input_hist.values(), np.sqrt(input_hist.variances()),
#                             ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
#                             ms=6, capsize=1, capthick=2, alpha=0.8)
#             main_ax.plot(x_range, out.eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
#                         label=fr"$\mu:{out.params['center'].value:.2f}, \sigma: {abs(out.params['sigma'].value):.2f}$")
#             main_ax.set_ylabel('Counts')
#             main_ax.set_ylim(-20, None)

#             main_ax.fill_between(
#                 x_range,
#                 out.eval(x=x_range) - dely,
#                 out.eval(x=x_range) + dely,
#                 color="hotpink",
#                 alpha=0.2,
#                 label='Uncertainty'
#             )
#             main_ax.legend(fontsize=18, loc='upper right')

#             subplot_ax.axvline(fit_min, c='red', lw=2)
#             subplot_ax.axvline(fit_max, c='red', lw=2)
#             subplot_ax.axhline(1, c='black', lw=0.75)
#             subplot_ax.axhline(0, c='black', lw=1.2)
#             subplot_ax.axhline(-1, c='black', lw=0.75)
#             subplot_ax.bar(centers, pulls, width=width, fc='royalblue')
#             subplot_ax.set_ylim(-2, 2)
#             subplot_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1])
#             subplot_ax.set_xlabel(r'Time Walk Corrected $\Delta$TOA [ps]')
#             subplot_ax.set_ylabel('Pulls', fontsize=20, loc='center')
#             subplot_ax.minorticks_off()

#             plt.tight_layout()

#         if (save):
#             fig.savefig(f'./lmfit_{chipNames}.png')
#             plt.close(fig)


#     if get_chisqure:
#         return [out.params['sigma'].value, out.params['sigma'].stderr, out.redchi]
#     else:
#         return [out.params['sigma'].value, out.params['sigma'].stderr]

## --------------------------------------