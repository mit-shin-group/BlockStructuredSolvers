using JLD2, CairoMakie
CairoMakie.activate!()

@load "kf_payload_256x1024x100.jld2" X_true X_hat
N = size(X_hat, 2)
t = 1:N
t = t ./ 10

# Style: role-based
col_true = :royalblue
col_est  = :crimson
lw_true  = 4.5
lw_est   = 3.0
alpha_true = 0.33   # soften the true curves
alpha_est  = 1.0   # estimates also slightly transparent

fig = Figure(size = (1600, 650), fontsize = 28)

# Top axis: x₁
ax1 = Axis(fig[1, 1];
    ylabel = L"x_1",
    ylabelsize = 30, yticklabelsize = 22,
    rightspinevisible = false, topspinevisible = false
)
hidexdecorations!(ax1, grid = false)

# Bottom axis: x₂ (carries shared x-axis)
ax2 = Axis(fig[2, 1];
    xlabel = "time step",
    ylabel = L"x_2",
    xlabelsize = 30, ylabelsize = 30,
    xticklabelsize = 22, yticklabelsize = 22,
    rightspinevisible = false, topspinevisible = false
)

linkxaxes!(ax1, ax2)

# x₁ traces
lines!(ax1, t, X_true[1, 2:end], color = (col_true, alpha_true), linewidth = lw_true)
lines!(ax1, t, X_hat[1, :],      color = (col_est, alpha_est),  linestyle = :dash, linewidth = lw_est)

# x₂ traces
lines!(ax2, t, X_true[2, 2:end], color = (col_true, alpha_true), linewidth = lw_true)
lines!(ax2, t, X_hat[2, :],      color = (col_est, alpha_est),  linestyle = :dash, linewidth = lw_est)

# Legend: role-based only
leg = Legend(fig,
    [LineElement(color = (col_true, alpha_true), linewidth = lw_true, linestyle = :solid),
     LineElement(color = (col_est,  alpha_est),  linewidth = lw_est,  linestyle = :dash)],
    ["true", "estimated"];
    orientation = :vertical, textsize = 26,
    framevisible = false, backgroundcolor = (:white, 0.85),
    patchsize = (32, 14)
)
fig[1:2, 2] = leg
colsize!(fig.layout, 2, Auto(0.13))

# Label(fig[0, 1], "Simulated Latent States", fontsize = 34, tellwidth = false)

save("kf_latents_role_colors_soft.pdf", fig)
println("Saved: kf_latents_role_colors_soft.pdf")


# using JLD2, CairoMakie
# CairoMakie.activate!()

# @load "kf_payload_256x1024x100.jld2" X_true X_hat

# N = size(X_hat, 2)
# t = 1:N

# fig = Figure(size = (1600, 650), fontsize = 28)

# # Top axis: x1 (hide x decorations so only bottom axis shows them)
# ax1 = Axis(fig[1, 1];
#     ylabel = L"x_1",
#     yticklabelsize = 22, ylabelsize = 28,
#     rightspinevisible = false, topspinevisible = false
# )
# hidexdecorations!(ax1, grid = false)

# # Bottom axis: x2 (this one carries the shared x-axis)
# ax2 = Axis(fig[2, 1];
#     xlabel = "time step",
#     ylabel = L"x_2",
#     xticklabelsize = 22, yticklabelsize = 22,
#     xlabelsize = 28, ylabelsize = 28,
#     rightspinevisible = false, topspinevisible = false
# )

# # Link x-axes so they share limits / zoom / pan
# linkxaxes!(ax1, ax2)

# # Colors
# c1, c2 = (:dodgerblue, :darkorange)

# # x1 traces
# l1 = lines!(ax1, t, X_true[1, 2:end], color = c1, linewidth = 3)
# l2 = lines!(ax1, t, X_hat[1, :],      color = c1, linestyle = :dash, linewidth = 3)

# # x2 traces
# l3 = lines!(ax2, t, X_true[2, 2:end], color = c2, linewidth = 3)
# l4 = lines!(ax2, t, X_hat[2, :],      color = c2, linestyle = :dash, linewidth = 3)

# # One shared legend on the right, spanning both rows
# leg = Legend(fig,
#     [l1, l2, l3, l4],
#     ["true x₁", "est. x₁", "true x₂", "est. x₂"];
#     orientation = :vertical, textsize = 24, framevisible = true, bgcolor = :white,
#     patchsize = (28, 12)
# )
# fig[1:2, 2] = leg
# colsize!(fig.layout, 2, Auto(0.15))   # give the legend a narrow column

# # Optional: common title
# Label(fig[0, 1], "Simulated Latent States", fontsize = 34, tellwidth = false)

# save("kf_latents_shared_x_single_legend.pdf", fig)
# println("Saved: kf_latents_shared_x_single_legend.pdf")

# ############## benchmark_kf_plot.jl ##############
# using JLD2, CairoMakie

# # Load saved arrays
# @load "kf_payload_256x1024x100.jld2" X_true X_hat n_state m_obs N_timesteps ρ σq σr

# N = size(X_hat, 2)
# tidx = 1:N

# CairoMakie.activate!()

# fig = Figure(size = (1600, 420), fontsize = 30)

# ax = Axis(fig[1, 1];
#     xlabel = "time step",
#     ylabel = L"x_1,\,x_2",
#     xlabelsize = 34,
#     ylabelsize = 36,
#     xticklabelsize = 26,
#     yticklabelsize = 26,
#     leftspinevisible   = true,
#     rightspinevisible  = false,
#     topspinevisible    = false,
#     bottomspinevisible = true
# )

# # Nice distinct color pairs for x1, x2
# col1 = :dodgerblue
# col2 = :darkorange

# # True trajectories: solid
# lines!(ax, tidx, X_true[1, 2:end], color = col1, linewidth = 3, label = "true x₁")
# lines!(ax, tidx, X_true[2, 2:end], color = col2, linewidth = 3, label = "true x₂")

# # Estimates: dashed, same colors
# lines!(ax, tidx, X_hat[1, :], color = col1, linestyle = :dash, linewidth = 3, label = "est x₁")
# lines!(ax, tidx, X_hat[2, :], color = col2, linestyle = :dash, linewidth = 3, label = "est x₂")

# # Legend: white background, larger text
# axislegend(ax; 
#     position = :rb, 
#     textsize = 28,
#     framevisible = true,
#     bgcolor = :white,
#     patchsize = (30, 14)
# )

# save("kf_latents_pretty.pdf", fig)
# println("Saved: kf_latents_pretty.pdf")
# ############ end ##############