# ============================================================
# χ²-Test Aufgabe – Lösung in R
# ============================================================

set.seed(42)
N <- 1000
p1 <- 0.25   # P(V1 = 1)
p2 <- 0.50   # P(V2 = 1)

# ============================================================
# Aufgabe 1: Simulation zweier unabhängiger binärer Variablen
# ============================================================

V1 <- rbinom(N, 1, p1)
V2 <- rbinom(N, 1, p2)

# Kontingenztabelle
a <- sum(V1 == 1 & V2 == 1)
b <- sum(V1 == 0 & V2 == 1)
c <- sum(V1 == 1 & V2 == 0)
d <- sum(V1 == 0 & V2 == 0)

tab1 <- matrix(c(a, c, b, d), nrow = 2,
               dimnames = list(V2 = c("1", "0"), V1 = c("1", "0")))
cat("=== Aufgabe 1: Simulierte Kontingenztabelle ===\n")
print(tab1)

test1 <- chisq.test(tab1, correct = FALSE)
cat("\nχ²-Test Ergebnis:\n")
print(test1)

cat("\nInterpretation: Da die Variablen unabhängig simuliert wurden,\n")
cat("erwarten wir einen großen p-Wert (> 0.05). Falls p < 0.05,\n")
cat("handelt es sich um einen Fehler 1. Art.\n\n")

# ============================================================
# Aufgabe 2: Perfekte Unabhängigkeit (Erwartungswerte)
# ============================================================

cat("=== Aufgabe 2: Tabelle unter perfekter Unabhängigkeit ===\n")

# Randsummen aus Aufgabe 1
rs_V1_1 <- a + c
rs_V1_0 <- b + d
rs_V2_1 <- a + b
rs_V2_0 <- c + d

# Erwartungswerte: E_ij = (Zeilensumme_i * Spaltensumme_j) / N
a2 <- round(rs_V2_1 * rs_V1_1 / N)
b2 <- round(rs_V2_1 * rs_V1_0 / N)
c2 <- round(rs_V2_0 * rs_V1_1 / N)
d2 <- round(rs_V2_0 * rs_V1_0 / N)

# Korrektur für Rundung
d2 <- N - a2 - b2 - c2

tab2 <- matrix(c(a2, c2, b2, d2), nrow = 2,
               dimnames = list(V2 = c("1", "0"), V1 = c("1", "0")))
print(tab2)

test2 <- chisq.test(tab2, correct = FALSE)
cat("\nχ²-Test Ergebnis:\n")
print(test2)

cat("\nErklärung: Wir modellieren hier perfekte Unabhängigkeit.\n")
cat("Jede Zelle entspricht genau dem Erwartungswert E_ij = (RS_i * RS_j)/N.\n")
cat("Da beobachtete = erwartete Häufigkeiten, ist χ² ≈ 0 und p ≈ 1.\n")
cat("→ Wir können H0 (Unabhängigkeit) NICHT ablehnen.\n\n")

# ============================================================
# Aufgabe 3 & 4: Gewicht auf die Hauptdiagonale verschieben
# ============================================================

cat("=== Aufgabe 3 & 4: Verschiebung auf Hauptdiagonale ===\n\n")
cat("In jedem Schritt: c → a (c-1, a+1) und b → d (b-1, d+1)\n")
cat("→ Modelliert zunehmende POSITIVE Abhängigkeit:\n")
cat("  V1=1 tritt häufiger gemeinsam mit V2=1 auf,\n")
cat("  V1=0 häufiger mit V2=0.\n\n")

max_steps <- min(b2, c2)
p_values <- numeric(max_steps + 1)
chi2_values <- numeric(max_steps + 1)

for (s in 0:max_steps) {
  tab_s <- matrix(c(a2 + s, c2 - s, b2 - s, d2 + s), nrow = 2)
  test_s <- chisq.test(tab_s, correct = FALSE)
  p_values[s + 1] <- test_s$p.value
  chi2_values[s + 1] <- test_s$statistic
}

# Ausgewählte Schritte anzeigen
cat(sprintf("%5s %5s %5s %5s %5s %10s %12s\n",
            "Step", "a", "b", "c", "d", "χ²", "p-Wert"))
cat(paste(rep("-", 58), collapse = ""), "\n")

show_steps <- unique(c(0, 1, 5, 10, 20, 50, max_steps))
for (s in show_steps) {
  cat(sprintf("%5d %5d %5d %5d %5d %10.4f %12.6f\n",
              s, a2 + s, b2 - s, c2 - s, d2 + s,
              chi2_values[s + 1], p_values[s + 1]))
}

# Schwelle finden
threshold <- which(p_values < 0.05)[1] - 1
cat(sprintf("\nErster Schritt mit p < 0.05: Schritt %d\n", threshold))
cat(sprintf("  χ² = %.4f, p = %.6f\n", chi2_values[threshold + 1], p_values[threshold + 1]))

cat("\nBeobachtung und Begründung:\n")
cat("Mit jedem Schritt wächst χ² monoton und der p-Wert sinkt.\n")
cat("Grund: Die Verschiebung auf die Diagonale erhöht die positive\n")
cat("Assoziation zwischen V1 und V2. Die beobachteten Häufigkeiten\n")
cat("weichen immer stärker von den unter Unabhängigkeit erwarteten\n")
cat("Häufigkeiten ab → χ² steigt → p-Wert fällt.\n")
cat("Ab einem bestimmten Punkt wird p < 0.05 und wir lehnen H0 ab.\n")

# ============================================================
# Plot
# ============================================================

par(mfrow = c(1, 2))

plot(0:max_steps, p_values, type = "l", col = "blue", lwd = 2,
     xlab = "Schritt", ylab = "p-Wert",
     main = "p-Wert vs. Diagonalverschiebung")
abline(h = 0.05, col = "red", lty = 2)
legend("topright", legend = "α = 0.05", col = "red", lty = 2)

plot(0:max_steps, chi2_values, type = "l", col = "darkgreen", lwd = 2,
     xlab = "Schritt", ylab = "χ²-Statistik",
     main = "χ²-Statistik vs. Diagonalverschiebung")
abline(h = qchisq(0.95, df = 1), col = "red", lty = 2)
legend("topleft", legend = "χ² kritisch (α=0.05)", col = "red", lty = 2)
